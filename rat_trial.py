from pydantic import BaseModel
from loguru import logger
from trial import Trial, Event
from enum import Enum
from pydantic import model_validator
import numpy as np
import os
import opensim as osim

class RatTrialType(str, Enum):
    STATIC = "Static"
    WALK = "Walk"

class RatTrial(Trial):
    required_markers: list[str] = [
        "TAIL", "SPL6", "LASI", "RASI", # Torso
        "LHIP", "LKNE", "LANK", "LTOE", # Left leg
        "RHIP", "RKNE", "RANK", "RTOE"  # Right leg
    ]
    
    required_parameters: list[str] = [
        "Mass",
        "Length",
        "RFemurLength",
        "RTibiaLength",
        "RFootLength",
        "LFemurLength",
        "LTibiaLength",
        "LFootLength",
    ]
    
    base_femur_length: float = float(np.linalg.norm([-0.0035000000000000001, -0.031199999999999999, -0.0050000000000000001]) * 1000)
    base_tibia_length: float = float(np.linalg.norm([0.0016000000000000001, 0.039, -0.0037000000000000002]) * 1000)
    
    @model_validator(mode='after')
    def _check_trial_type(self):
        if not self.trial_type:
            for trial_type in RatTrialType:
                if trial_type.value.lower() in self.name.lower():
                    self.trial_type = trial_type
                    break
            else:
                raise ValueError("Trial type must be specified if not inferrable from name")
        return self
    
    def validate_trial(self) -> bool:
        if self.trial_type == RatTrialType.STATIC:
            return self.valid_static()
        elif self.trial_type == RatTrialType.WALK:
            return self.valid_walk()
        return False
    
    def valid_static(self) -> bool:
        """
        For rat trials, a valid static trial will have:
         - At least one frame with all required markers
            - OpenSim takes the average of all frames to calculate marker distances for scaling
         - Parameters needed for scaling
        """
        # There must be at least one frame with all required markers
        if not self.find_full_frames(self.required_markers):
            logger.info(f"Trial {self.name} has no frames with all required markers")
            return False
        for parameter in self.required_parameters:
            if parameter not in self.parameters:
                logger.info(f"Trial {self.name} is missing required parameter {parameter}")
                return False
        return True

    def valid_walk(self) -> bool:
        """
        For rat trials, a valid walk trial will have:
         - 7 events in frame / time order:
            - Lead Foot Strike
            - Opposite Foot Off
            - Opposite Foot Strike
            - Lead Foot Off
            - Lead Foot Strike
            - Opposite Foot Off
            - Opposite Foot Strike
         - Markers for every frame in events
         - Force plate contexts labeled for left and right
            - If > 1 force plate labeled for either left or right, must be consecutive (?)
        """
        # Check order of events
        if len(self.events) != 7:
            logger.info(f"Trial {self.name} has {len(self.events)} events, expected 7")
            return False
        # Check for correct context + label order (Events should already be in order by frame/time)
        lead_context = self.events[0].context
        opposite_context = "Right" if lead_context == "Left" else "Left"
        expected_order = [
            (lead_context, "Foot Strike"),
            (opposite_context, "Foot Off"),
            (opposite_context, "Foot Strike"),
            (lead_context, "Foot Off"),
            (lead_context, "Foot Strike"),
            (opposite_context, "Foot Off"),
            (opposite_context, "Foot Strike"),
        ]        
        for event, expected in zip(self.events, expected_order):
            if (event.context, event.label) != expected:
                logger.info(f"Trial {self.name} has events out of order or with incorrect context/label")
                return False
        # Check for required markers for every frame between first and last events
        first_event = self.events[0].get_frame(self.points.rate) or 0
        last_event = self.events[-1].get_frame(self.points.rate) or self.points.total_frames
        if self.check_point_gaps(self.required_markers, regions = [(first_event, last_event)]):
            logger.info(f"Trial {self.name} has gaps in required markers between events")
            return False
        
        # Check for force plate contexts labeled for left and right
        
        return True

    def get_stance_phases(self, side: str) -> list[tuple[Event, Event]]:
        """
        Get the stance phase for a specific side.
        Stance phase is defined as the time between foot strike and foot off events for that side.
        """
        stance_phases = []
        foot_strike = None
        foot_off = None
        for event in self.events:
            if event.context == side:
                if event.label == "Foot Strike":
                    foot_strike = event
                elif event.label == "Foot Off" and foot_strike:
                    foot_off = event
            if foot_strike and foot_off:
                stance_phases.append((foot_strike, foot_off))
                foot_strike = None
                foot_off = None
        return stance_phases
    
    def get_swing_phases(self, side: str) -> list[tuple[Event, Event]]:
        """
        Get the swing phase for a specific side.
        Swing phase is defined as the time between foot off and next foot strike events for that side.
        """
        swing_phases = []
        foot_off = None
        next_foot_strike = None
        for event in self.events:
            if event.context == side:
                if event.label == "Foot Off":
                    foot_off = event
                elif event.label == "Foot Strike" and foot_off:
                    next_foot_strike = event
            if foot_off and next_foot_strike:
                swing_phases.append((foot_off, next_foot_strike))
                foot_off = None
                next_foot_strike = None
        return swing_phases
    
    def get_stance_swing_phases(self, side: str) -> list[tuple[Event, Event, Event]]:
        """
        Get the coupled stance and swing phases for a specific side.
        Each tuple contains (foot strike, foot off, next foot strike).
        """
        stance_swing_phases = []
        foot_strike = None
        foot_off = None
        next_foot_strike = None
        for event in self.events:
            if event.context == side:
                if event.label == "Foot Strike" and not foot_strike:
                    foot_strike = event
                elif event.label == "Foot Off" and foot_strike:
                    foot_off = event
                elif event.label == "Foot Strike" and foot_off:
                    next_foot_strike = event
            if foot_strike and foot_off and next_foot_strike:
                stance_swing_phases.append((foot_strike, foot_off, next_foot_strike))
                foot_strike = None
                foot_off = None
                next_foot_strike = None
        return stance_swing_phases
        

    # Spatiotemporal parameters  -- Currently following Huxham et al. 2006 for straight line gait
    # TODO: Implement Dingwell 2024 calculations
    def stride_time(self)->dict[str,list[float]]:
        """
        """
        times = {"Left": [], "Right": []}
        for side in ["Left", "Right"]:
            foot_strike = self.get_events(label="Foot Strike", context=side)
            if not foot_strike:
                logger.warning(f"No foot strike events found for {side} side")
                continue
            foot_strike_times = [event.get_time(self.points.rate) for event in foot_strike]
            if len(foot_strike_times) < 2:
                logger.warning(f"Not enough foot strike events for {side} side to calculate stride time")
                continue
            for i in range(len(foot_strike_times) - 1):
                start_time = foot_strike_times[i]
                end_time = foot_strike_times[i + 1]
                time_diff = end_time - start_time
                times[side].append(time_diff)
        return times
    
    def stride_length(self)->dict[str,list[float]]:
        lengths = {"Left": [], "Right": []}
        for side in ["Left", "Right"]:
            foot_marker = side[0].upper() + "TOE"
            if foot_marker not in self.points.trajectories:
                logger.warning(f"Marker {foot_marker} not found in trial points")
                continue
            foot_strike = self.get_events(label="Foot Strike", context=side)
            if not foot_strike:
                logger.warning(f"No foot strike events found for {side} side")
                continue
            foot_strike_frames = [event.get_frame(self.points.rate) for event in foot_strike]
            if len(foot_strike_frames) < 2:
                logger.warning(f"Not enough foot strike events for {side} side to calculate stride length")
                continue
            for i in range(len(foot_strike_frames) - 1):
                start_frame = foot_strike_frames[i]
                end_frame = foot_strike_frames[i + 1]
                start_pos = self.points.get_marker_coords(foot_marker, start_frame)
                end_pos = self.points.get_marker_coords(foot_marker, end_frame)
                if start_pos is None or end_pos is None:
                    logger.warning(f"Missing marker position for {foot_marker} at frames {start_frame} or {end_frame}")
                    continue
                length = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
                lengths[side].append(length)
        return lengths

    def stride_width(self):
        raise NotImplementedError("Stride width calculation is not implemented yet.")

    def stride_velocity(self)->dict[str,list[float]]:
        stride_lengths = self.stride_length()
        stride_times = self.stride_time()
        velocities = {"Left": [], "Right": []}

        for side in ["Left", "Right"]:
            lengths = stride_lengths.get(side, [])
            times = stride_times.get(side, [])
            
            if not lengths or not times:
                logger.warning(f"Not enough data to calculate stride velocity for {side} side")
                continue

            # Ensure we have a 1:1 mapping of lengths to times
            # This assumes that stride_length and stride_time will return lists of the same length for a given side
            # and that the i-th length corresponds to the i-th time.
            for i in range(min(len(lengths), len(times))):
                if times[i] == 0: # Avoid division by zero
                    logger.warning(f"Stride time is zero for {side} side, cannot calculate velocity for stride {i}")
                    velocities[side].append(float('nan')) # Or handle as appropriate
                else:
                    velocities[side].append(lengths[i] / times[i])
            
            if len(lengths) != len(times):
                logger.warning(f"Mismatch in number of stride lengths and times for {side} side. Velocity calculated for {min(len(lengths), len(times))} strides.")

        return velocities

    def stride_cadence(self)->dict[str,list[float]]:
        """
        """
        velocities = self.stride_velocity()
        cadences = {"Left": [1/vel for vel in velocities.get("Left", []) if vel != 0],
                    "Right": [1/vel for vel in velocities.get("Right", []) if vel != 0]}
        return cadences

    def step_time(self) -> dict[str, list[float]]:
        """
        Step time is calculated as the time from foot strike to next contralateral foot strike
        """
        left_foot_strikes = self.get_events(label="Foot Strike", context="Left")
        right_foot_strikes = self.get_events(label="Foot Strike", context="Right")
        times = {"Left": [], "Right": []}
        
        
        return times

    def step_length(self):
        """
        """
        pass

    def step_width(self)->list[float]:
        """
        """
        widths = []
        return widths
    
    def step_velocity(self):
        pass
    
    def step_cadence(self)->list[float]:
        """
        """
        cadences=[]
        return cadences

    def stance_time(self) -> dict[str, list[float]]:
        """
        Calculates the stance time for each stride as the time between foot strike and foot off events for one side
        """
        times = {"Left": [], "Right": []}
        for side in ["Left", "Right"]:
            foot_strike = self.get_events(label="Foot Strike", context=side)
            foot_off = self.get_events(label="Foot Off", context=side)
            if not foot_strike:
                logger.warning(f"No foot strike events found for {side} side")
                continue
            if not foot_off:
                logger.warning(f"No foot off events found for {side} side")
                continue
            foot_strike_times = [event.get_time(self.points.rate) for event in foot_strike]
            foot_off_times = [event.get_time(self.points.rate) for event in foot_off]
            
    def stance_percentage(self) -> dict[str, list[float]]:
        """
        Calculates the percentage of stance time for each stride 
        """
        percentages = {"Left": [], "Right": []}
        stance_times = self.stance_time()
        stride_times = self.stride_time()
        for side in ["Left", "Right"]:
            stance = stance_times.get(side, [])
            stride = stride_times.get(side, [])
            if not stance or not stride:
                logger.warning(f"Not enough data to calculate stance percentage for {side} side")
                continue
            for i in range(min(len(stance), len(stride))):
                if stride[i] == 0:
                    logger.warning(f"Stride time is zero for {side} side, cannot calculate stance percentage for stride {i}")
                    percentages[side].append(float('nan'))
                else:
                    percentage = (stance[i] / stride[i]) * 100
                    percentages[side].append(percentage)
        return percentages
            
    # Body mass and inertia properties
    def thigh_mass(self):
        mass = self.parameters["Mass"]
        return (7.3313*mass+3.6883)/1000

    def thigh_com(self, side: str) -> tuple[float, float, float]:
        side = side.capitalize()
        if side not in ["L", "R"]:
            raise ValueError("Side must be 'L' or 'R'")
        femur_length = self.parameters[f"{side}FemurLength"]
        mass = self.parameters["Mass"]
        return (
            femur_length*(-8.7844332/100000), 
            mass*0.148741316*(-42.118041/100), 
            mass*0.098448042*(2.00427791/100) * -1 if side == "L" else 1
        )
    
    def thigh_moi(self, side: str) -> tuple[float, float, float]:
        side = side.capitalize()
        if side not in ["L", "R"]:
            raise ValueError("Side must be 'L' or 'R'")
        femur_length = self.parameters[f"{side}FemurLength"]
        mass = self.parameters["Mass"]
        return (
            (0.00189568)*(mass)*(femur_length/1000)**2,
            (0.00143871)*(mass)*(femur_length/1000)**2,
            (0.00248006)*(mass)*(femur_length/1000)**2
        )

    def shank_mass(self):
        mass = self.parameters["Mass"]
        return (3.2096*mass+3.0047)/1000
    
    def shank_com(self, side: str) -> tuple[float, float, float]:
        side = side.capitalize()
        if side not in ["L", "R"]:
            raise ValueError("Side must be 'L' or 'R'")
        tibia_length = self.parameters[f"{side}TibiaLength"]
        mass = self.parameters["Mass"]
        return (
            (mass)*0.09004923*(-2.43352222/100), 
            tibia_length*(67.363643/100000), 
            (mass*0.07731125)*(1.71207065/100) * -1 if side == "L" else 1
        )
        
    def shank_moi(self, side: str) -> tuple[float, float, float]:
        side = side.capitalize()
        if side not in ["L", "R"]:
            raise ValueError("Side must be 'L' or 'R'")
        tibia_length = self.parameters[f"{side}TibiaLength"]
        mass = self.parameters["Mass"]
        return (
            (0.00104229)*(mass)*(tibia_length/1000)**2,
            (0.00029337)*(mass)*(tibia_length/1000)**2,
            (0.00104734)*(mass)*(tibia_length/1000)**2
        )

    def foot_mass(self):
        mass = self.parameters["Mass"]
        return (2.2061*mass+0.87788)/1000
    
    def foot_com(self, side: str) -> tuple[float, float, float]:
        side = side.capitalize()
        if side not in ["L", "R"]:
            raise ValueError("Side must be 'L' or 'R'")
        foot_length = self.parameters[f"{side}FootLength"]
        mass = self.parameters["Mass"]
        return (
            (mass*0.04627387)*(-4.294993/100),
            foot_length*(-42.78009/100000),
            (mass*0.07246637)*(0.6265934/100) * -1 if side == "L" else 1
        ) # TODO: Still need to check the weird thing Brody does with this in the old code
        
    def foot_moi(self, side: str) -> tuple[float, float, float]:
        side = side.capitalize()
        if side not in ["L", "R"]:
            raise ValueError("Side must be 'L' or 'R'")
        foot_length = self.parameters[f"{side}FootLength"]
        mass = self.parameters["Mass"]
        return (
            (0.000384786)*(mass)*(foot_length/1000)**2,
            (0.0000518802)*(mass)*(foot_length/1000)**2,
            (0.000364591)*(mass)*(foot_length/1000)**2
        )

    def scale_opensim_model(self,                             
                            unscaled_model_path: str, 
                            marker_set_path: str, 
                            marker_file_name: str,
                            output_dir: str = '.', 
                            scale_setup_path: str | None = None
                            ):
        """
        Create scaled OpenSim models (one with markers moved, one without) from a static rat trial.
        Args:
            unscaled_model_path (str): Path to the unscaled OpenSim model file (.osim).
            marker_set_path (str): Path to the marker set file (.xml).
            marker_file_name (str): Name of the marker file to be used for scaling. Needs to be in the same directory as the unscaled model.
            output_dir (str): Directory where the scaled models and scale setup will be saved.
            scale_setup_path (str | None): Path to an existing scale setup file. If None, a new one will be created.
        
        Note: OpenSim's path handling is trash and inconsistent
        """        
        unscaled_model_path = os.path.abspath(unscaled_model_path)
        marker_set_path = os.path.abspath(marker_set_path)
        output_dir = os.path.abspath(output_dir)
        marker_file_name = os.path.basename(marker_file_name) # Ensure we only use the file name, not the path
        
        if scale_setup_path is not None and os.path.exists(scale_setup_path):
            scale_tool = osim.ScaleTool(os.path.abspath(scale_setup_path))
        else:
            scale_tool = osim.ScaleTool()
        scale_tool.setName(self.name)
        
        model_scaler: osim.ModelScaler = scale_tool.getModelScaler()
        model_scaler.setApply(True)
        scaled_model_path = os.path.join(output_dir, f"{self.name}_scaled.osim")
        model_scaler.setOutputModelFileName(scaled_model_path)
        scale_factors_path = os.path.join(output_dir, f"{self.name}_scale.xml")
        model_scaler.setOutputScaleFileName(scale_factors_path)
        model_scaler.setMarkerFileName(marker_file_name)

        time_range = osim.ArrayDouble()
        first_time = self.points.time_from_frame(self.points.first_frame)
        last_time = self.points.time_from_frame(self.points.last_frame)
        time_range.set(0, first_time)
        time_range.set(1, last_time)
        model_scaler.setTimeRange(time_range)

        scale_tool.setSubjectMass(self.parameters["Mass"])
        
        # Manual scaling factors - This is probably the only thing before run that cannot be abstracted out
        scale_set: osim.ScaleSet = model_scaler.getScaleSet()
        scale_set.get(0).setScaleFactors(osim.Vec3(self.parameters["RFemurLength"]/self.base_femur_length))
        scale_set.get(1).setScaleFactors(osim.Vec3(self.parameters["RTibiaLength"]/self.base_tibia_length))
        scale_set.get(2).setScaleFactors(osim.Vec3(self.parameters["LFemurLength"]/self.base_femur_length))
        scale_set.get(3).setScaleFactors(osim.Vec3(self.parameters["LTibiaLength"]/self.base_tibia_length))
        
        marker_placer: osim.MarkerPlacer = scale_tool.getMarkerPlacer()
        marker_placer.setApply(True)
        marker_model_name = f"{self.name}_marker.osim"
        marker_placer.setOutputModelFileName(marker_model_name)
        marker_placer.setMarkerFileName(marker_file_name)
        marker_placer.setTimeRange(time_range)
        
        generic_model_maker: osim.GenericModelMaker = scale_tool.getGenericModelMaker()
        generic_model_maker.setModelFileName(unscaled_model_path)
        generic_model_maker.setMarkerSetFileName(marker_set_path)

        scale_setup_path = os.path.join(output_dir, f"{self.name}_scale_setup.xml")
        scale_tool.printToXML(scale_setup_path)
        self.link_file("scale_setup", scale_setup_path)
        
        scale_tool = osim.ScaleTool(scale_setup_path) # I don't think this is necessary, but it seems to be MAMP convention

        scale_tool.run()
        self.link_file("scale_factors", scale_factors_path)
        
        
        scaled_model = osim.Model(scaled_model_path)
        scaled_model.setName(scaled_model_path.replace(".osim", ""))

        marker_model_path = os.path.join(output_dir, marker_model_name)
        marker_model = osim.Model(marker_model_path)
        marker_model.setName(marker_model_name.replace(".osim", ""))
        
        for model in [scaled_model, marker_model]:
            for side in ["L", "R"]:
                side_short = side[0].lower()
                model_body_set: osim.BodySet = model.getBodySet()
                
                thigh: osim.Body = model_body_set.get(f"femur_{side_short}")
                thigh.set_mass(self.thigh_mass())
                thigh.set_mass_center(osim.Vec3(*self.thigh_com(side)))
                thigh.set_inertia(osim.Vec6(*self.thigh_moi(side), 0, 0, 0))

                shank: osim.Body = model_body_set.get(f"tibia_{side_short}")
                shank.set_mass(self.shank_mass())
                shank.set_mass_center(osim.Vec3(*self.shank_com(side)))
                shank.set_inertia(osim.Vec6(*self.shank_moi(side), 0, 0, 0))

                foot: osim.Body = model_body_set.get(f"foot_{side_short}")
                foot.set_mass(self.foot_mass())
                foot.set_mass_center(osim.Vec3(*self.foot_com(side)))
                foot.set_inertia(osim.Vec6(*self.foot_moi(side), 0, 0, 0))
            out_path = os.path.join(output_dir, model.getName() + ".osim")
            model.printToXML(out_path)
        self.link_file("scaled_model", scaled_model_path)
        self.link_file("marker_model", marker_model_path)

    
    # def create_scaled_mjcf(self,
    #                         unscaled_model_path: str,
    #                         output_path: str | None = None
    #                         ):
    #     if self.trial_type != "Static" or not self.valid_static():
    #         logger.warn("Trial is not a valid static trial")
    #     import mujoco as mjc
    #     spec = mjc.MjSpec.from_file(unscaled_model_path)
        
    #     for side in ['Left', 'Right']:
    #         femur = f"femur_{side[0].lower()}"
    #         femur = spec.body(f"{femur}")
    #         femur.mass = self.thigh_mass()
    #         femur.ipos = self.thigh_com(side[0])                
    #         femur.inertia = list(self.thigh_moi(side[0]))
            
    #         tibia = f"tibia_{side[0].lower()}"
    #         tibia = spec.body(f"{tibia}")
    #         tibia.mass = self.shank_mass()
    #         tibia.ipos = self.shank_com(side[0])
    #         tibia.inertia = list(self.shank_moi(side[0]))
            
    #         foot = f"foot_{side[0].lower()}"
    #         foot = spec.body(f"{foot}")
    #         foot.mass = self.foot_mass()
    #         foot.ipos = self.foot_com(side[0])
    #         foot.inertia = list(self.foot_moi(side[0]))
        
        
    #     self.mjcf_model = spec.compile()
    #     spec.to_xml()
    
    
import polars as pl
class RatSession(BaseModel):
    """
    Represents a session of rat trials.
    """
    name: str
    path_stem: str
    static_trials: list[RatTrial] = []
    static_trial: RatTrial | None = None
    walk_trials: list[RatTrial] = []
    
    @model_validator(mode='after')
    def check_trials(self):
        if not self.static_trials:
            raise ValueError("Session must contain at least one static trial")
        # Iterate backwards until we find a valid static trial since later Static trials are more likely to be used
        for trial in reversed(self.static_trials):
            if trial.valid_static():
                self.static_trial = trial
                logger.info(f"Using static trial {trial.name} for OpenSim analysis")
                break
        else:
            raise ValueError("No valid static trial found for OpenSim analysis")
        if not self.walk_trials:
            logger.warning("Session has no walk trials, some analyses may not be applicable")
        return self
       
