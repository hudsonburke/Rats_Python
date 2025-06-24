import ezc3d
from pydantic import BaseModel, model_validator, field_validator
from enum import Enum
from typing import Any, Type, TypeVar
import numpy as np
from loguru import logger
import polars as pl
import os
import opensim as osim
from collections import defaultdict

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
    
    @property 
    def time(self) -> np.ndarray:
        """
        Return a time vector for the time series group.
        """
        return np.arange(self.first_frame, self.last_frame + 1) / self.rate

class MarkerTrajectory(BaseModel):
    """
    A marker trajectory represented as a Polars DataFrame with columns:
    x, y, z, residual, description
    """
    class Config:
        arbitrary_types_allowed = True
    data: pl.DataFrame
    description: str = ''
    
    def __init__(self, **kwargs):
        if 'data' in kwargs:
            data = kwargs['data']
        else:
            data = pl.DataFrame({
                'x': kwargs.get('x', []),
                'y': kwargs.get('y', []),
                'z': kwargs.get('z', []),
                'residual': kwargs.get('residual', []),
            })
        description = kwargs.get('description', '')
        super().__init__(data=data, description=description)
            
    
    @field_validator('data')
    @classmethod
    def validate_dataframe_structure(cls, v: pl.DataFrame) -> pl.DataFrame:
        """Validate that the DataFrame has the required columns"""
        required_columns = ['x', 'y', 'z', 'residual']
        
        if not all(col in v.columns for col in required_columns):
            missing = [col for col in required_columns if col not in v.columns]
            raise ValueError(f"Missing required columns: {missing}")
        
        # Ensure correct data types
        try:
            v = v.with_columns([
                pl.col('x').cast(pl.Float64),
                pl.col('y').cast(pl.Float64),
                pl.col('z').cast(pl.Float64),
                pl.col('residual').cast(pl.Float64),
            ])
        except Exception as e:
            raise ValueError(f"Error casting columns to correct types: {e}")
        return v
    
    @property
    def coords(self) -> np.ndarray:
        """Return coordinates as numpy array (n_frames, 3)"""
        return self.data.select(['x', 'y', 'z']).to_numpy()
    
    @property
    def residual(self) -> np.ndarray:
        """Return residuals as numpy array (n_frames,)"""
        return self.data.select(['residual']).to_numpy().flatten()

    def __len__(self) -> int:
        return len(self.data)
    
    def rename_columns(self, prefix: str) -> pl.DataFrame:
        """Rename columns with a prefix for concatenation"""
        return self.data.rename({
            'x': f'{prefix}_x',
            'y': f'{prefix}_y', 
            'z': f'{prefix}_z',
            'residual': f'{prefix}_residual',
        })
    
class Points(TimeSeriesGroup):
    units: str
    trajectories: dict[str, MarkerTrajectory]
    
    @model_validator(mode='after')
    def validate_trajectory_lengths(self) -> 'Points':
        """Ensure all trajectories have the same length matching total_frames"""
        expected_length = self.total_frames
        
        for marker_name, trajectory in self.trajectories.items():
            if len(trajectory) != expected_length:
                raise ValueError(
                    f"Marker '{marker_name}' has {len(trajectory)} frames, "
                    f"expected {expected_length} frames"
                )
        return self
    
    def to_df(self) -> pl.DataFrame:
        """
        Convert the Points object to a Polars DataFrame.
        Each marker's coordinates will be separate columns (marker_x, marker_y, marker_z, marker_residual).
        """
        if not self.trajectories:
            return pl.DataFrame()
        
        dfs = []
        for name, trajectory in self.trajectories.items():
            marker_df = trajectory.rename_columns(name)
            dfs.append(marker_df)
        
        # Add time column
        time_df = pl.DataFrame({
            'time': self.time
        })
        
        # Concatenate horizontally
        all_dfs = [time_df] + dfs
        return pl.concat(all_dfs, how='horizontal')
    
    def get_marker_coords(self, marker_name: str, frame: int | None = None) -> np.ndarray:
        """Get marker coordinates, optionally at a specific frame"""
        if marker_name not in self.trajectories:
            raise ValueError(f"Marker '{marker_name}' not found in trajectories")        
        marker = self.trajectories[marker_name]
        
        if frame is None:
            return marker.coords
        
        if frame < self.first_frame or frame > self.last_frame:
            raise IndexError(f"Frame {frame} out of bounds")

        # Convert absolute frame to relative index
        frame_idx = frame - self.first_frame
        return marker.coords[frame_idx]
    
    def add_marker(self, name: str, x: list, y: list, z: list, 
                   residual: list | None = None, description: str = ''):
        """Add a new marker trajectory"""
        n_frames = self.total_frames
        
        # Validate lengths
        if len(x) != n_frames or len(y) != n_frames or len(z) != n_frames:
            raise ValueError(f"Coordinate arrays must have length {n_frames}")
        
        if residual is None:
            residual = [0.0] * n_frames
            
        trajectory = MarkerTrajectory(
            data = pl.DataFrame({
                'x': x,
                'y': y,
                'z': z,
                'residual': residual
            }),
            description=description
        )
        self.trajectories[name] = trajectory
    
class AnalogChannel(BaseModel):
    """Each analog channel can have different units"""
    units: str
    data: list[float]
    description: str = ''
    scale: float = 1.0
    offset: float = 0.0
    
class EZC3DForcePlatform(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        
    unit_force: str = 'N'  # Default force unit
    unit_moment: str = 'Nm'  # Default moment unit
    unit_position: str = 'm'  # Default position unit
    cal_matrix: np.ndarray = np.eye(6)  # Calibration matrix for force platform
    corners: np.ndarray = np.zeros((3, 4))  # 4 corners in 3D space
    origin: np.ndarray = np.zeros(3)  # Origin of the force platform
    data: pl.DataFrame = pl.DataFrame()  # Data for the force platform
    # Moments and center of pressure are expressed in global
    
    @field_validator('cal_matrix')
    @classmethod
    def validate_cal_matrix(cls, v: np.ndarray) -> np.ndarray:
        """Validate that the calibration matrix is 6x6"""
        if v.shape != (6, 6):
            raise ValueError("Calibration matrix must be 6x6")
        return v
    
    @field_validator('corners')
    @classmethod
    def validate_corners(cls, v: np.ndarray) -> np.ndarray:
        """Validate that the corners are a 3x4 array"""
        if v.shape != (3, 4):
            raise ValueError("Corners must be a 3x4 array")
        return v
    
    @field_validator('origin')
    @classmethod
    def validate_origin(cls, v: np.ndarray) -> np.ndarray:
        """Validate that the origin is a 3D vector"""
        if v.shape != (3,):
            raise ValueError("Origin must be a 3D vector")
        return v
    
    @field_validator('data')
    @classmethod
    def validate_data_structure(cls, v: pl.DataFrame) -> pl.DataFrame:
        """Validate that the DataFrame has the required columns"""
        required_columns = ['force_x', 'force_y', 'force_z', 
                            'moment_x', 'moment_y', 'moment_z', 
                            'center_of_pressure_x', 'center_of_pressure_y', 'center_of_pressure_z', 
                            'free_moment_x', 'free_moment_y', 'free_moment_z']

        if not all(col in v.columns for col in required_columns):
            missing = [col for col in required_columns if col not in v.columns]
            raise ValueError(f"Missing required columns: {missing}")
        
        # Ensure correct data types
        try:
            v = v.with_columns([
                pl.col('force_x').cast(pl.Float64),
                pl.col('force_y').cast(pl.Float64),
                pl.col('force_z').cast(pl.Float64),
                pl.col('moment_x').cast(pl.Float64),
                pl.col('moment_y').cast(pl.Float64),
                pl.col('moment_z').cast(pl.Float64),
                pl.col('center_of_pressure_x').cast(pl.Float64),
                pl.col('center_of_pressure_y').cast(pl.Float64),
                pl.col('center_of_pressure_z').cast(pl.Float64),
                pl.col('free_moment_x').cast(pl.Float64),
                pl.col('free_moment_y').cast(pl.Float64),
                pl.col('free_moment_z').cast(pl.Float64)                
            ])
        except Exception as e:
            raise ValueError(f"Error casting columns to correct types: {e}")
        return v
    
    @property
    def force(self) -> np.ndarray:
        """Return force as a numpy array (n_frames, 3)"""
        return self.data.select(['force_x', 'force_y', 'force_z']).to_numpy()
    
    @property
    def moment(self) -> np.ndarray:
        """Return moment as a numpy array (n_frames, 3)"""
        return self.data.select(['moment_x', 'moment_y', 'moment_z']).to_numpy()
    
    @property
    def center_of_pressure(self) -> np.ndarray:
        """Return center of pressure as a numpy array (n_frames, 3)"""
        return self.data.select(['center_of_pressure_x', 'center_of_pressure_y', 'center_of_pressure_z']).to_numpy()
    
    @property
    def free_moment(self) -> np.ndarray:
        """Return free moment as a numpy array (n_frames, 3)"""
        return self.data.select(['free_moment_x', 'free_moment_y', 'free_moment_z']).to_numpy()


class Analogs(TimeSeriesGroup):
    # Analogs store different channels each of which could have different units
    channels: dict[str, AnalogChannel]
    gen_scale: float = 1.0 # General scale factor for all channels
    
    def to_df(self) -> pl.DataFrame:
        """
        Convert the Analogs object to a Polars DataFrame.
        Each channel will be a column in the DataFrame.
        WARNING: This decouples the channels from their original units.
        """
        if not self.channels:
            return pl.DataFrame()
        dfs = []
        for name, channel in self.channels.items():
            channel_df = pl.DataFrame({
                name: channel.data
            })
            dfs.append(channel_df)
        
        # Concatenate horizontally
        return pl.concat(dfs, how='horizontal')

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
    classification: str = ''
    trial_type: str | None = None
    import_method: ImportMethod
    linked_files: dict[str, str] = {} # Map of associated files, e.g. C3D file path, etc.
    parameters: dict[str, Any] = {}

    events: list[Event] # Should be in ascending order by frame or time

    points: Points
    point_gaps: dict[str, list[tuple[int, int]]] = {}
    
    analogs: Analogs
    force_platforms: list[EZC3DForcePlatform] = [] # List of force platforms, if any
    
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
    
    def to_sto(self,
               filepath: str,
               rotation: np.ndarray = np.eye(3),
               ):
        """
        Export the analog data to OpenSim .sto file format.
        """
        raise DeprecationWarning("The to_sto method is deprecated. Use the opensim.STOFileAdapter directly instead.")
        import opensim as osim
        table = osim.TimeSeriesTable()
        table.addTableMetaDataString("nColumns", str(len(self.analogs.channels)))
        table.addTableMetaDataString("nRows", str(self.analogs.last_frame - self.analogs.first_frame + 1))
        table.setColumnLabels(list(self.analogs.channels.keys()))
        time_col = self.analogs.time
        analogs_df = self.analogs.to_df()
        # Rotate the coordinates
        analogs_df = analogs_df.with_columns(pl.Series('time', time_col))
        for row in analogs_df.iter_rows(named=True):
            time = row['time']
            values = [row[col] for col in analogs_df.columns if col != 'time']
            table.appendRow(time, osim.RowVector(values))
        adapter = osim.STOFileAdapter()
        adapter.write(table, filepath)
        self.link_file('sto', filepath)

    def to_trc(self, 
               filepath: str,
               output_units: str | None = None,
               rotation: np.ndarray = np.eye(3)
               ):
        """
        Export the marker data to TRC file format used by OpenSim
        """  
        import opensim as osim
        table = osim.TimeSeriesTableVec3()
        markers = list(self.points.trajectories.keys())
        table.setColumnLabels(markers)
        conversion_factor = 1.0
        if output_units is not None and self.points.units != output_units:
            logger.info(f"Output units {output_units} do not match points units {self.points.units}. Converting coordinates.")
            # TODO: Convert coordinates to the desired output units
            
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
            
        subject_names = cls._get_c3d_param(c3d_object, 'SUBJECTS', 'NAMES', default=[])

        # FORCE_PLATFORM - Using ezc3d's force platform filter
        c3d_force_platforms = c3d_data['platform']
        force_platforms = []  # Reset force platforms
        for fp in c3d_force_platforms:
            # Convert to EZC3DForcePlatform
            force = fp.get('force', [[]])
            moment = fp.get('moment', [[]])
            position = fp.get('center_of_pressure', [[]])
            free_moment = fp.get('Tz', [])
            ezc3d_fp = EZC3DForcePlatform(
                unit_force=fp.get('unit_force', 'N'),
                unit_moment=fp.get('unit_moment', 'Nm'),
                unit_position=fp.get('unit_position', 'm'),
                cal_matrix=np.array(fp.get('cal_matrix', np.eye(6))),
                corners=np.array(fp.get('corners', np.zeros((4, 3)))),
                origin=np.array(fp.get('origin', np.zeros(3))),
                data=pl.DataFrame({
                    'force_x': force[0, :],
                    'force_y': force[1, :],
                    'force_z': force[2, :],
                    'moment_x': moment[0, :],
                    'moment_y': moment[1, :],
                    'moment_z': moment[2, :],
                    'center_of_pressure_x': position[0, :],
                    'center_of_pressure_y': position[1, :],
                    'center_of_pressure_z': position[2, :],
                    'free_moment_x': free_moment[0, :],
                    'free_moment_y': free_moment[1, :],
                    'free_moment_z': free_moment[2, :]
                })
            )
            force_platforms.append(ezc3d_fp)

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
        point_units = cls._get_c3d_param(c3d_object, 'POINT', 'UNITS', default=['mm'])[0]
        
        # if point_scale != 1: # TODO: Figure out what to do with this
        #     logger.warning(f"Point scale {point_scale} is not 1. Scaling point data accordingly.")
        #     point_data = point_data * point_scale
        for i, label in enumerate(point_labels):
            # Create MarkerTrajectory from the data
            n_frames = point_data.shape[2]            
            trajectory = MarkerTrajectory(
                x=point_data[0, i, :].tolist(),
                y=point_data[1, i, :].tolist(), 
                z=point_data[2, i, :].tolist(),
                residual=residuals[0, i, :].tolist(),
                description=point_descriptions[i]
            )
            trajectories[label] = trajectory
    
        # Analogs
        channels = {}
        analog_rate_param = cls._get_c3d_param(c3d_object, 'ANALOG', 'RATE', default=None)
        if analog_rate_param != analogs_rate:
            logger.warning(f"Analog rate {analog_rate_param} does not match header rate {analogs_rate}")
        analog_data = c3d_data['analogs'] # 1xMxP (data, labels, num_frames)
        analog_units = cls._get_c3d_param(c3d_object, 'ANALOG', 'UNITS', default=[])    
        analog_descriptions = cls._get_c3d_param(c3d_object, 'ANALOG', 'DESCRIPTIONS', default=[])    
        analog_labels = cls._get_c3d_param(c3d_object, 'ANALOG', 'LABELS', default=[])
        analog_offsets = cls._get_c3d_param(c3d_object, 'ANALOG', 'OFFSET', default=np.zeros(len(analog_labels)))
        analog_scales = cls._get_c3d_param(c3d_object, 'ANALOG', 'SCALE', default=np.ones(len(analog_labels)))
        analog_gen_scale = cls._get_c3d_param(c3d_object, 'ANALOG', 'GEN_SCALE', default=[1.0])[0]
        
        for i, label in enumerate(analog_labels):
            channels[label] = AnalogChannel(
                data=analog_data[0, i, :].tolist(),  # Convert to list for compatibility
                units=analog_units[i] or '',
                description=analog_descriptions[i] or '',
                scale=analog_scales[i],
                offset=analog_offsets[i]
            )    
        return cls(
            name=trial_name,
            session_name=session_name,
            subject_names=subject_names,
            classification=classification,
            import_method=ImportMethod.C3D,
            parameters=parameters,
            events=events,
            points=Points(
                first_frame=points_first_frame,
                last_frame=points_last_frame,
                rate=points_rate,
                units=point_units,
                trajectories=trajectories
            ),
            analogs=Analogs(
                first_frame=analogs_first_frame,
                last_frame=analogs_last_frame,
                rate=analogs_rate,
                channels=channels,
                gen_scale=analog_gen_scale,
            ),
            force_platforms=force_platforms,
        )
           
    @classmethod
    def from_c3d_file(cls: Type[_T], file_path: str) -> _T:
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
                               output_dir: str, 
                               applied_bodies: dict[int, str], # Expected to be a dict of {platform_index: body_name} where platform_index is 1-based
                               force_expressed_in_body: str = 'ground',
                               point_expressed_in_body: str = 'ground', 
                               force_identifier: str = r'force%d_v',
                               point_identifier: str = r'force%d_p',
                               torque_identifier: str = r'moment%d_',
                               rotation: np.ndarray = np.eye(3),
                               mot_filename: str | None = None,
                               external_loads_filename: str | None = None,
                               unit_force: str = 'N',
                               unit_position: str = 'm',
                               unit_moment: str = 'Nm',
                               metadata: dict[str, Any] = {}
                               ):
        """
        Export force plate metadata to OpenSim ExternalLoads .xml file and the data to a .mot file.
        
        For now, extract foot contact from .enf files, but in the future use contact 
        """
        ext_loads = osim.ExternalLoads()

        if external_loads_filename is None:
            external_loads_filename = f"{self.name}_fp_setup.xml"
        external_loads_filepath = os.path.join(output_dir, external_loads_filename)
        
        if mot_filename is None:
            mot_filename = f"{self.name}_FP.mot"
        mot_filepath = os.path.join(output_dir, mot_filename)
        mot_labels = []
        mot_table = osim.TimeSeriesTable()
        time_col = self.analogs.time

        data = np.zeros((len(self.force_platforms)*9, len(time_col)))  # 3 forces, 3 moments, 3 center of pressure
        for i, fp in enumerate(self.force_platforms): 
            display_i = i + 1  # For display purposes, OpenSim uses 1-based indexing
            # Create ExternalForce for each force platform        
            ext_force = osim.ExternalForce()
            ext_force.setName(f'FP{str(display_i)}')
            ext_force.setAppliedToBodyName(applied_bodies[display_i] if display_i in applied_bodies else '')

            ext_force.setForceExpressedInBodyName(force_expressed_in_body)
            fp_force_identifier = force_identifier % (display_i)
            ext_force.setForceIdentifier(fp_force_identifier)

            ext_force.setPointExpressedInBodyName(point_expressed_in_body)
            fp_point_identifier = point_identifier % (display_i)
            ext_force.setPointIdentifier(fp_point_identifier)

            fp_torque_identifier = torque_identifier % (display_i)
            ext_force.setTorqueIdentifier(fp_torque_identifier)
            
            ext_force.set_data_source_name(mot_filename) 
            
            ext_loads.cloneAndAppend(ext_force)
            
            # Determine conversion factors for units
            force_conversion_factor, position_conversion_factor, moment_conversion_factor = 1.0, 1.0, 1.0
            if fp.unit_force != unit_force:
                logger.warning(f"Force platform {display_i} force unit {fp.unit_force} does not match output unit {unit_force}. Converting forces.")
                current_units = osim.Units(fp.unit_force)
                force_conversion_factor = current_units.convertTo(osim.Units(unit_force))
            if fp.unit_position != unit_position:
                logger.warning(f"Force platform {display_i} position unit {fp.unit_position} does not match output unit {unit_position}. Converting positions.")
                current_units = osim.Units(fp.unit_position)
                position_conversion_factor = current_units.convertTo(osim.Units(unit_position))
            if fp.unit_moment != unit_moment:
                logger.warning(f"Force platform {display_i} torque unit {fp.unit_moment} does not match output unit {unit_moment}. Converting torques.")
                current_units = osim.Units(fp.unit_moment)
                moment_conversion_factor = current_units.convertTo(osim.Units(unit_moment))
                
            # Rotate the force platform data
            force = np.array(rotation @ np.array(fp.force).T).T * force_conversion_factor * -1.0 # OpenSim expects forces to be in the opposite direction
            cop = np.array(rotation @ np.array(fp.center_of_pressure).T).T * position_conversion_factor 
            free_moment = np.array(rotation @ np.array(fp.free_moment).T).T * moment_conversion_factor * -1.0 # OpenSim expects moments to be in the opposite direction
            mot_labels.extend([fp_force_identifier + coord for coord in 'xyz'])
            mot_labels.extend([fp_torque_identifier + coord for coord in 'xyz'])
            mot_labels.extend([fp_point_identifier + coord for coord in 'xyz'])  # This could be precomputed, but having it next to the data makes it clear what order it should be added
            data[i*9:i*9+3, :] = force.T  # 3 forces
            data[i*9+3:i*9+6, :] = free_moment.T  # 3 moments
            data[i*9+6:i*9+9, :] = cop.T  # 3 center of pressure
        for i in range(len(time_col)):
            mot_table.appendRow(time_col[i], osim.RowVector(data[:, i]))

        print(f"Writing {len(mot_labels)} labels to MOT file: {mot_labels}")
        mot_table.setColumnLabels(mot_labels)

        for key, value in metadata.items():
            mot_table.addTableMetaDataString(key, str(value))

        if 'nRows' not in metadata:
            n_frames = self.force_platforms[0].data.height # Assuming all platforms have the same number of frames
            mot_table.addTableMetaDataString('nRows', str(n_frames))
        if 'nColumns' not in metadata:
            n_columns = len(self.force_platforms) * 9  # 3 forces, 3 moments, 3 center of pressure
            mot_table.addTableMetaDataString('nColumns', str(n_columns))
        adapter = osim.STOFileAdapter()
        adapter.write(mot_table, mot_filepath)
        ext_loads.setDataFileName(mot_filename)
        ext_loads.printToXML(external_loads_filepath)

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
