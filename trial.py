import ezc3d
from pydantic import BaseModel, model_validator, ConfigDict
from enum import Enum
from abc import ABC, abstractmethod 
from typing import Any, Annotated, Type, TypeVar 
import numpy as np
from loguru import logger
import polars as pl
import patito as pt
import os
import opensim as osim

# Define a TypeVar that is bound by the Trial class itself
# This means _T can be Trial or any subclass of Trial.
_T = TypeVar("_T", bound="Trial") # TODO: Replace with 3.12+ Generic types

# Inspiration from https://medium.com/the-pythonworld/never-use-none-for-missing-values-again-do-this-instead-8a92e20b6954
class Sentinel:
    def __init__(self, name: str):
        self.name = name
    def __repr__(self):
        return f"<{self.name}>"

MISSING = Sentinel("MISSING")
MISSING_LIST = [] 
UNSET = Sentinel("UNSET") 

class Event(BaseModel): 
    """
    Times will default to being stored in seconds.
    See c3d event specification for details.
    """
    label: str
    context: str
    frame: int | None = None
    time: float | None = None
    description: str | None = None
    
    @model_validator(mode='after')
    def validate_frames_or_times(self):
        assert self.frame is not None or self.time is not None, "Either frames or times must be provided."
        assert self.frame is None or self.time is None, "Only one of frames or times should be provided."
        return self

    def get_frame(self, point_rate: float | None) -> int:
        if self.frame is not None:
            return self.frame
        if self.time is not None and point_rate is not None and point_rate > 0:
            return int(self.time * point_rate)
        # This should not happen if validate_frames_or_times is called first
        raise ValueError("Cannot compute frame without point rate or time.")
        
    def get_time(self, point_rate: float | None) -> float:
        if self.time is not None:
            return self.time
        if self.frame is not None and point_rate is not None and point_rate > 0:
            return self.frame / point_rate
        # This should not happen if validate_frames_or_times is called first
        raise ValueError("Cannot compute time without point rate or frame.")
    
class ImportMethod(str, Enum):
    C3D = "C3D"
    VICON_NEXUS = "Vicon Nexus"
    CUSTOM = "Custom"

class TimeSeriesGroup(BaseModel):
    first_frame: int
    last_frame: int
    rate: float
    
    @model_validator(mode='after')
    def validate_frames_and_rate(self):
        """
        Validate that first_frame is non-zero and less than last_frame and rate is positive.
        """
        assert self.first_frame >= 0, "first_frame must be non-negative"
        assert self.first_frame < self.last_frame, "first_frame must be less than last_frame"
        assert self.rate > 0, "rate must be positive"
        return self
    
    @property
    def total_frames(self):
        return self.last_frame - self.first_frame + 1
    
    def time_from_frame(self, frame: int) -> float:
        if frame < self.first_frame or frame > self.last_frame:
            raise ValueError("Frame out of bounds")
        return frame / self.rate

class TimeSeries(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    description: str | None = None
    data: pl.DataFrame 

    @property
    @abstractmethod
    def data_schema(self) -> type[pt.Model]:
        """Abstract property for the Patito schema of the channel\'s DataFrame."""
        raise NotImplementedError
    
    @model_validator(mode='after')
    def validate_dataframe_schema(self):
        if self.data is not None:
            # Validate the DataFrame against the schema provided by data_schema
            self.data_schema.validate(self.data)
        return self

class MarkerTrajectory(TimeSeries):
    class _MarkerDataFrameSchema(pt.Model):
        x: float
        y: float
        z: float
        # Pulling from c3d, you get residuals, but Vicon Nexus just gives you 'exists'
        residual: float | None = None 
        exists: bool | None = None
        
    @property
    def data_schema(self) -> type[pt.Model]:
        return self._MarkerDataFrameSchema
    
    @property
    def coords(self) -> np.ndarray:
        return self.data.select(['x', 'y', 'z']).to_numpy() if self.data is not None else np.array([])

class Points(TimeSeriesGroup):
    units: str # All points must have the same units
    trajectories: dict[str, MarkerTrajectory]
    
    def get_marker_coords(self, marker_name: str, frame: int | None = None) -> np.ndarray | None:
        if marker_name not in self.trajectories:
            return None
        marker = self.trajectories[marker_name]
        if frame is None:
            return marker.coords
        if frame < self.first_frame or frame > self.last_frame:
            return None
        # Get the row corresponding to the frame
        row_index = frame - self.first_frame
        if row_index < 0 or row_index >= len(marker.data):
            return None
        return marker.coords[row_index]
    
class Analog(TimeSeries):
    units: str
    class _AnalogDataFrameSchema(pt.Model):
        values: float
        
    @property
    def data_schema(self) -> type[pt.Model]:
        return self._AnalogDataFrameSchema

class ForcePlate(Analog):
    context: str
    local_r: Annotated[np.ndarray, '3x3 local rotation matrix'] = np.eye(3) #TODO: Maybe use numpydantic
    local_t: Annotated[np.ndarray, '3x1 local translation vector'] = np.zeros(3)
    world_r: Annotated[np.ndarray, '3x3 world rotation matrix'] = np.eye(3)
    world_t: Annotated[np.ndarray, '3x1 world translation vector'] = np.zeros(3)
    lower_bounds: Annotated[np.ndarray, '3x1 lower bounds'] = np.array([-np.inf, -np.inf, -np.inf])
    upper_bounds: Annotated[np.ndarray, '3x1 upper bounds'] = np.array([np.inf, np.inf, np.inf])

class Analogs(TimeSeriesGroup):
    # Analogs store different channels each of which could have different units
    channels: dict[str, Analog]

class OpenSimOutput(str, Enum):
    """
    Enum for OpenSim output types.
    """
    SCALED_MODEL = "scaled_model"
    MARKER_MODEL = "marker_model"
    TRC = "trc"
    IK_SETUP = "ik_setup"
    IK = "ik_results"
    ID_SETUP = "id_setup"
    ID = "id_results"

class Trial(BaseModel):
    
    # Trial Metadata
    name: str 
    session_name: str | None
    subject_names: list[str] | str | None 
    classification: str | None = None
    trial_type: str | None = None
    import_method: ImportMethod
    linked_files: dict[str, str] = {} # Map of associated files, e.g. C3D file path, etc.
    
    parameters: dict[str, Any] = {}

    events: list[Event] # Should be in ascending order by frame or time

    points: Points
    point_gaps: dict[str, list[tuple[int, int]]] | None = None
    
    analogs: Analogs
    
    def get_events(self, label: str = "", context: str = "") -> list[Event]:
        """
        Return a copy of the events list filtered by label and context.
        If label or context is empty, it will not filter by that parameter.
        """
        return [
            event for event in self.events
            if (not label or event.label == label) and (not context or event.context == context)
        ]
    
    @model_validator(mode='after')
    def order_events(self) -> 'Trial':
        """
        Ensure events are in ascending order by frame or time.
        """
        self.events = sorted(self.events, key=lambda e: (e.get_frame(self.points.rate), e.get_time(self.points.rate)))
        return self
        
    def link_file(self, file_key:str, file_path:str):
        self.linked_files[file_key] = os.path.abspath(file_path)
    
    def get_linked_file(self, file_key: str) -> str | None:
        """
        Get the absolute path of an associated file by its key.
        Returns None if the file is not associated.
        """
        return self.linked_files.get(file_key, None)
    
    def to_trc(self, 
               filepath: str,
               output_units: str | None = None,
               rotation: np.ndarray = np.eye(3)
               ):
        """
        Export the trial data to TRC file format used by OpenSim
        """  
        import opensim as osim
        table = osim.TimeSeriesTableVec3()
        markers = list(self.points.trajectories.keys())
        table.setColumnLabels(markers)
        conversion_factor = 1.0
        if output_units is not None and self.points.units != output_units:
            logger.info(f"Output units {output_units} do not match points units {self.points.units}. Converting coordinates.")
            # Convert coordinates to the desired output units
            
        table.addTableMetaDataString("Units", self.points.units)
        table.addTableMetaDataString("DataRate", str(self.points.rate))
        for frame in range(self.points.first_frame, self.points.last_frame + 1):
            row = []
            for marker_name in markers:
                in_coords = self.points.get_marker_coords(marker_name, frame)
                if in_coords is not None:
                    coords = np.array(rotation @ np.array(in_coords).T).T   # Apply rotation if needed
                    coords = coords * conversion_factor # Convert coordinates if needed
                    row.append(osim.Vec3(coords[0], coords[1], coords[2]))
                else:
                    row.append(osim.Vec3().setToNaN())
            time = self.points.time_from_frame(frame)
            table.appendRow(time, osim.RowVectorVec3(row))
        adapter = osim.TRCFileAdapter()
        # Make sure the directories exist
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        adapter.write(table, filepath)
        self.link_file('trc', filepath)

    @staticmethod
    def _get_c3d_param(c3d_object: ezc3d.c3d, *keys, default=None):
        """
        Helper function to get nested parameters from a C3D object.
        """
        param = c3d_object.parameters
        for key in keys:
            param = param.get(key, {})
        return param.get('value', default)

    @classmethod
    def from_c3d(cls: Type[_T], 
                 c3d_object: ezc3d.c3d, 
                 trial_name: str = "", 
                 session_name: str = "", 
                 classification: str = "") -> _T: 
        """
        Create a Trial instance from a C3D object.
        """
        c3d_header = c3d_object.header
        c3d_parameters = c3d_object.parameters
        c3d_data = c3d_object.data

        # Header
        points_rate = c3d_header['points']['frame_rate']
        points_first_frame = c3d_header['points']['first_frame']
        points_last_frame = c3d_header['points']['last_frame']
        
        analogs_rate = c3d_header['analogs']['frame_rate']
        analogs_first_frame = c3d_header['analogs']['first_frame']
        analogs_last_frame = c3d_header['analogs']['last_frame']
        
        ## Parameters
        # TRIAL
        camera_rate = cls._get_c3d_param(c3d_object, 'TRIAL', 'CAMERA_RATE', default=None)
        if camera_rate != points_rate:
            logger.warning(f"Camera rate {camera_rate} does not match points rate {points_rate} in header")

        # FORCE_PLATFORM - # TODO
        # EVENT_CONTEXT - not currently using
        # EVENT
        events = []
        if 'EVENT' in c3d_parameters:
            event_params : dict = c3d_parameters['EVENT']
            num_events = event_params.get('USED', {}).get('value', [0])[0]
            event_contexts = event_params.get('CONTEXTS', {}).get('value', [])
            event_labels = event_params.get('LABELS', {}).get('value', [])
            event_descriptions = event_params.get('DESCRIPTIONS', {}).get('value', [])
            # event_subjects = event_params.get('SUBJECTS', {}).get('value', [])
            event_times = event_params.get('TIMES', {}).get('value', [])
            for i in range(num_events):
                events.append(Event(
                    label=event_labels[i],
                    context=event_contexts[i],
                    time=event_times[0][i]*60 + event_times[1][i], # Convert from (min, sec) to sec
                    description=event_descriptions[i],
                ))
                
        # MANUFACTURER - not currently using
        # ANALYSIS - These are mainly STPs from Vicon Nexus, so I think I'll just compute myself 
        # also because they don't really follow the convention of the c3d format
        
        # PROCESSING
        parameters = {}
        if 'PROCESSING' in c3d_parameters:
            for key, value in c3d_parameters['PROCESSING'].items():
                arr = value.get('value', None)
                if arr is not None and len(arr) == 1:
                    parameters[key] = arr[0]
                else:
                    parameters[key] = arr
                
        ## Data 
        # Points
        trajectories = {}
        point_rate_param = cls._get_c3d_param(c3d_object, 'POINT', 'RATE', default=None)
        if point_rate_param != points_rate:
            logger.warning(f"Point rate {point_rate_param} does not match header rate {points_rate}")
        point_data = c3d_data['points'] # 4xNxM (XYZ1, labels, num_frames)
        residuals = c3d_data['meta_points']['residuals'] # 1xNxM
        point_labels = cls._get_c3d_param(c3d_object, 'POINT', 'LABELS', default=[])
        point_descriptions = cls._get_c3d_param(c3d_object, 'POINT', 'DESCRIPTIONS', default=[])
        point_scale = cls._get_c3d_param(c3d_object, 'POINT', 'SCALE', default=1)
        
        # if point_scale != 1: # TODO: Figure out what to do with this
        #     logger.warning(f"Point scale {point_scale} is not 1. Scaling point data accordingly.")
        #     point_data = point_data * point_scale
        for i, label in enumerate(point_labels):
            trajectories[label] = MarkerTrajectory(
                data=pl.DataFrame({
                'x': point_data[0, i, :],
                'y': point_data[1, i, :],
                'z': point_data[2, i, :],
                'residual': residuals[0, i, :] if residuals is not None else None,
                'exists': point_data[3, i, :] == 1
                }),
                description=point_descriptions[i] or None
            )
        # Analogs
        channels = {}
        analog_rate_param = cls._get_c3d_param(c3d_object, 'ANALOG', 'RATE', default=None)
        if analog_rate_param != analogs_rate:
            logger.warning(f"Analog rate {analog_rate_param} does not match header rate {analogs_rate}")
        analog_data = c3d_data['analogs'] # 1xMxP (data, labels, num_frames)
        analog_units = cls._get_c3d_param(c3d_object, 'ANALOG', 'UNITS', default=[])    
        analog_descriptions = cls._get_c3d_param(c3d_object, 'ANALOG', 'DESCRIPTIONS', default=[])    
        analog_labels = cls._get_c3d_param(c3d_object, 'ANALOG', 'LABELS', default=[])
        # TODO: if ezc3d doesn't handle it, will need to deal with GAIN, SCALE, OFFSET
        
        for i, label in enumerate(analog_labels):
            channels[label] = Analog(
                data=pl.DataFrame({
                    'values': analog_data[0, i, :]
                }),
                units=analog_units[i] or 'unknown',
                description=analog_descriptions[i] or None
            )    
        # Force platforms - # TODO
        return cls(
            name=trial_name,
            session_name=session_name,
            subject_names=cls._get_c3d_param(c3d_object, 'SUBJECTS', 'NAMES', default=[]),
            classification=classification,
            import_method=ImportMethod.C3D,
            parameters=parameters,
            events=events,
            points = Points(
                first_frame=points_first_frame,
                last_frame=points_last_frame,
                rate=points_rate,
                units=c3d_parameters.get('POINTS', {}).get('UNITS', {}).get('value', 'mm'),
                trajectories=trajectories
            ),
            analogs=Analogs(
                first_frame=analogs_first_frame,
                last_frame=analogs_last_frame,
                rate=analogs_rate,
                channels=channels
            )
        )
           
    @classmethod
    def from_c3d_file(cls: Type[_T], file_path: str) -> _T: # MODIFIED: Returns an instance of _T
        """
        Create a Trial instance from a C3D file.
        """
        import os
        # Check that the file exists and is a valid C3D file
        file_path = os.path.normpath(file_path)
        if not file_path.endswith('.c3d'):
            raise ValueError("File must be a C3D file.")
        try:
            c3d_object = ezc3d.c3d(file_path, extract_forceplat_data=True)
        except Exception as e:
            logger.error(f"Failed to read C3D file {file_path}: {e}")
            c3d_object = ezc3d.c3d()
            # raise ValueError(f"Invalid C3D file: {file_path}") from e
        split_path = file_path.split(os.sep)
        trial_name = split_path[-1].replace('.c3d', '')
        session_name = split_path[-2] if len(split_path) > 1 else ""
        classification = split_path[-4] if len(split_path) > 3 else ""
        trial = cls.from_c3d(c3d_object, trial_name=trial_name, session_name=session_name, classification=classification)
        trial.link_file("c3d", file_path)
        return trial 

    @classmethod
    def from_vicon_nexus(cls) -> 'Trial':
        """
        Create a Trial instance from an open trial in Vicon Nexus.
        This method requires the Vicon Nexus API to be installed and configured.
        https://pycgm2.readthedocs.io/en/latest/Pages/thirdParty/NexusAPI.html
        
        """ 
        raise NotImplementedError("Vicon Nexus API integration is not implemented yet.")

    def check_point_gaps(self, 
            marker_names : list[str] | None = None, 
            regions: list[tuple[int,int] | tuple[float, float]] | None= None
        ) -> dict[str, list[tuple[int, int]]]:
        """
        Check for gaps in point data for specified markers and regions.
        A gap is defined as any frame in the region where the marker data is missing (NaN).
        Returns a dictionary with marker names as keys and lists of (start, end) tuples indicating integer frame gaps.
        
        If no markers or regions are specified, checks all markers and the entire trial duration.
        If already computed, return the cached result.
        """
        
        if self.point_gaps is not None:
            # Check for markers and regions already computed
            relevant_gaps = {}
            for marker, gap_list in self.point_gaps.items():
                if marker not in relevant_gaps:
                    relevant_gaps[marker] = []
                relevant_gaps[marker].extend(gap_list)
            return relevant_gaps
        
        gaps = {}
        if marker_names is None:
            marker_names = list(self.points.trajectories.keys())
        if regions is None:
            regions = [(self.points.first_frame, self.points.last_frame)] 

        for region in regions:
            start, end = region
            if isinstance(start, float):
                start = int(start * self.points.rate)
            if isinstance(end, float):
                end = int(end * self.points.rate)
            for marker in marker_names:
                if marker not in self.points.trajectories:
                    gaps[marker] = [(start, end)]
                    continue
                marker_data = self.points.trajectories[marker].data
                # Check if marker data exists in every frame in the region
                region_data = marker_data[start:end+1]
                for coord in ['x', 'y', 'z']:
                    missing_data = region_data.filter(pl.col(coord).is_null())
                    if not missing_data.is_empty():
                        if marker not in gaps:
                            gaps[marker] = []
                        gaps[marker].append((start, end))
                        break         
        return gaps
    
    @model_validator(mode='after')
    def _cache_point_gaps(self) -> 'Trial':
        self.point_gaps = self.check_point_gaps()
        return self
 
    def find_full_frames(self, marker_names: list[str] | None = None) -> list[int]:
        """
        Find all frames where all specified markers have data.
        If no markers are specified, checks all markers.
        Returns a list of frame indices.
        """
        if marker_names is None:
            marker_names = list(self.points.trajectories.keys())
        full_frames = set(range(self.points.first_frame, self.points.last_frame + 1))
        for marker in marker_names:
            if marker not in self.points.trajectories:
                return []
            marker_data = self.points.trajectories[marker].data
            marker_full_frames = set(marker_data.filter(
                (pl.col('x').is_not_null()) & 
                (pl.col('y').is_not_null()) & 
                (pl.col('z').is_not_null())
            ).select(pl.arange(0, marker_data.height)).to_series().to_list())
            full_frames &= marker_full_frames
        return sorted(full_frames)
    
    # TODO Refactor out opensim tools to own submodule? Trial should be agnostic of opensim model
    
    def run_opensim_ik(self, 
                        model_path: str, 
                        trc_path: str | None = None,
                        output_dir: str = '.',
                        start_time: float = 0.0,
                        end_time: float = np.inf, 
                        ik_setup_path: str | None = None
                        ):
        if ik_setup_path is None:
            ik_tool = osim.InverseKinematicsTool()
        else:
            ik_tool = osim.InverseKinematicsTool(os.path.abspath(ik_setup_path))

        ik_tool.setName(self.name)
        model = osim.Model(os.path.abspath(model_path))
        ik_tool.setModel(model)
        ik_tool.setMarkerDataFileName(f"{self.name}.trc" if not trc_path else trc_path)
        ik_results_name = f"{self.name}_ik.mot"
        ik_results_path = os.path.join(output_dir, ik_results_name)
        ik_tool.setOutputMotionFileName(ik_results_path)
        ik_tool.setResultsDir(os.path.abspath(output_dir))
        
        # TODO: Could be pulled from trc
        ik_tool.setStartTime(start_time)
        ik_tool.setEndTime(end_time)
        
        ik_setup_path = os.path.join(output_dir, f'{self.name}_ik_setup.xml')
        ik_tool.printToXML(ik_setup_path)
        self.link_file(OpenSimOutput.IK_SETUP, ik_setup_path)
        ik_tool.run()
        self.link_file(OpenSimOutput.IK, ik_results_path)

    def export_force_platforms(self, 
                               filepath: str, 
                               force_platforms: list[ForcePlate],
        ):
        """
        Export force plate metadata to OpenSim ExternalLoads .xml file and the data to a .sto file.
        """
        ext_loads = osim.ExternalLoads()
        ext_loads.setDataFileName(f"{self.name}_external_loads.sto")
        
        for fp in force_platforms:
            ext_force = osim.ExternalForce()
            ext_force.setName('')
            ext_force.setAppliedToBodyName('')
            
            ext_force.setForceExpressedInBodyName('')
            ext_force.setForceIdentifier('')
            
            ext_force.setPointExpressedInBodyName('')
            ext_force.setPointIdentifier('')
            
            ext_force.setTorqueIdentifier('')
            
            ext_force.set_data_source_name('') # relative 
            
            ext_loads.cloneAndAppend(ext_force)

    def run_opensim_id(self, 
                        model_path: str, 
                        ik_results_path: str | None = None,
                        output_dir: str = '.',
                        id_setup_path: str | None = None
                        ):
        if id_setup_path is None:
            id_tool = osim.InverseDynamicsTool()
        else:
            id_tool = osim.InverseDynamicsTool(os.path.abspath(id_setup_path))

        id_tool.setName(self.name)
        model = osim.Model(os.path.abspath(model_path))
        id_tool.setModel(model)
        
        ik_results_path = ik_results_path or self.get_linked_file('ik_results')
        id_tool.setCoordinatesFileName(ik_results_path)
        
        
        # TODO: Maintain relative paths for IK Setup files
        #   - Set paths relative to the trial directory?
        #   - Could always set working directory to the trial directory
        #   - OR print with relative paths and then set the tool to use absolute paths (see MATLAB toolbox)
        
        id_results_name = f"{self.name}_id.mot"
        id_results_path = os.path.join(output_dir, id_results_name)
        id_tool.setOutputGenForceFileName(id_results_path)
        id_tool.setResultsDir(os.path.abspath(output_dir))
