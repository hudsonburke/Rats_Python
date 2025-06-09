from ..Toolbox_Python.trial import Trial
from enum import Enum
from pydantic import model_validator
from loguru import logger

class RatTrialType(str, Enum):
    STATIC = "Static"
    WALK = "Walk"

class RatTrial(Trial):
    
    required_markers = [
        "TAIL", "SPL6", "LASI", "RASI", # Torso
        "LHIP", "LKNE", "LANK", "LTOE", # Left leg
        "RHIP", "RKNE", "RANK", "RTOE"  # Right leg
    ]
    
    required_parameters = [
        "Mass",
        "Length",
        "RightFemurLength",
        "RightTibiaLength",
        "RightFootLength",
        "LeftFemurLength",
        "LeftTibiaLength",
        "LeftFootLength",
    ]
        
    
    @model_validator(mode='after')
    def _check_trial_type(self):
        if not self.trial_type:
            for trial_type in RatTrialType:
                if trial_type.value in self.name:
                    self.trial_type = trial_type
                    break
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
         - No gap filling in history (?) # TODO
        """
        for marker in self.required_markers:
            # TODO: Check that at least one frame has all markers
            if marker not in self.points.coords['markers']:
                return False
        for parameter in self.required_parameters:
            if parameter not in self.parameters:
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
        first_event = self.events[0].get_frame(self.point_rate) or 0
        last_event = self.events[-1].get_frame(self.point_rate) or self.total_frames - 1
        if self.check_point_gaps(self.required_markers, regions = [(first_event, last_event)]):
            logger.info(f"Trial {self.name} has gaps in required markers between events")
            return False
        return True

    def rat_thigh_mass(self):
        mass = self.parameters["Mass"].value
        return (7.3313*mass+3.6883)/1000

    def rat_thigh_com(self, side: str) -> tuple[float, float, float]:
        side = side.capitalize()
        if side not in ["Left", "Right"]:
            raise ValueError("Side must be 'Left' or 'Right'")
        femur_length = self.parameters[f"{side}FemurLength"].value
        mass = self.parameters["Mass"].value
        return (
            femur_length*(-8.7844332/100000), 
            mass*0.148741316*(-42.118041/100), 
            mass*0.098448042*(2.00427791/100)
        )
    
    def rat_thigh_moi(self, side: str) -> tuple[float, float, float]:
        side = side.capitalize()
        if side not in ["Left", "Right"]:
            raise ValueError("Side must be 'Left' or 'Right'")
        femur_length = self.parameters[f"{side}FemurLength"].value
        mass = self.parameters["Mass"].value
        return (
            (0.00189568)*(mass)*(femur_length/1000)**2,
            (0.00143871)*(mass)*(femur_length/1000)**2,
            (0.00248006)*(mass)*(femur_length/1000)**2
        )

    def rat_shank_mass(self):
        mass = self.parameters["Mass"].value
        return (3.2096*mass+3.0047)/1000
    
    def rat_shank_com(self, side: str) -> tuple[float, float, float]:
        side = side.capitalize()
        if side not in ["Left", "Right"]:
            raise ValueError("Side must be 'Left' or 'Right'")
        tibia_length = self.parameters[f"{side}TibiaLength"].value
        mass = self.parameters["Mass"].value
        return (
            (mass)*0.09004923*(-2.43352222/100), 
            tibia_length*(67.363643/100000), 
            (mass*0.07731125)*(1.71207065/100)
        )
        
    def rat_shank_moi(self, side: str) -> tuple[float, float, float]:
        side = side.capitalize()
        if side not in ["Left", "Right"]:
            raise ValueError("Side must be 'Left' or 'Right'")
        tibia_length = self.parameters[f"{side}TibiaLength"].value
        mass = self.parameters["Mass"].value
        return (
            (0.00104229)*(mass)*(tibia_length/1000)**2,
            (0.00029337)*(mass)*(tibia_length/1000)**2,
            (0.00104734)*(mass)*(tibia_length/1000)**2
        )

    def rat_foot_mass(self):
        mass = self.parameters["Mass"].value
        return (2.2061*mass+0.87788)/1000
    
    def rat_foot_com(self, side: str) -> tuple[float, float, float]:
        side = side.capitalize()
        if side not in ["Left", "Right"]:
            raise ValueError("Side must be 'Left' or 'Right'")
        foot_length = self.parameters[f"{side}FootLength"].value
        mass = self.parameters["Mass"].value
        return (
            (mass*0.04627387)*(-4.294993/100),
            foot_length*(-42.78009/100000),
            (mass*0.07246637)*(0.6265934/100)
        )
        
    def rat_foot_moi(self, side: str) -> tuple[float, float, float]:
        side = side.capitalize()
        if side not in ["Left", "Right"]:
            raise ValueError("Side must be 'Left' or 'Right'")
        foot_length = self.parameters[f"{side}FootLength"].value
        mass = self.parameters["Mass"].value
        return (
            (0.000384786)*(mass)*(foot_length/1000)**2,
            (0.0000518802)*(mass)*(foot_length/1000)**2,
            (0.000364591)*(mass)*(foot_length/1000)**2
        )

    def create_scaled_model(self, move_markers: bool = True):
        import opensim as osim
        if self.trial_type != "Static" or not self.valid_static():
            raise ValueError("Trial is not a valid static trial")
        
        scale_tool = osim.ScaleTool()
        scale_tool.setName(self.name)
        
        model_scaler : osim.ModelScaler = scale_tool.getModelScaler()
        model_scaler.setApply(True)
        
    def inverse_kinematics(self, output_path: str | None = None):
        import opensim as osim
        
        if output_path is None:
            logger.warning("No output path specified for inverse kinematics, using working directory")
            output_path = f"{self.name}_ik.mot"
        
        ik_tool = osim.InverseKinematicsTool()
        ik_tool.setName(self.name)
        ik_tool.setModel(self.model)

    def inverse_dynamics(self, output_path: str | None = None):
        import opensim as osim

        if output_path is None:
            logger.warning("No output path specified for inverse dynamics, using working directory")
            output_path = f"{self.name}_id.mot"

        id_tool = osim.InverseDynamicsTool()
        id_tool.setName(self.name)
        id_tool.setModel(self.model)