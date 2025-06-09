from ..Toolbox_Python.trial import Trial
from enum import Enum
from pydantic import model_validator
from loguru import logger
import numpy as np


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
    
    base_femur_length = np.linalg.norm([-0.0035000000000000001, -0.031199999999999999, -0.0050000000000000001])*1000
    base_tibia_length = np.linalg.norm([0.0016000000000000001, 0.039, -0.0037000000000000002])*1000

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
        """
        # There must be at least one frame with all required markers
        if not self.find_full_frames(self.required_markers):
            logger.info(f"Trial {self.name} has no frames with all required markers")
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
        first_event = self.events[0].get_frame(self.points.rate) or 0
        last_event = self.events[-1].get_frame(self.points.rate) or self.points.total_frames
        if self.check_point_gaps(self.required_markers, regions = [(first_event, last_event)]):
            logger.info(f"Trial {self.name} has gaps in required markers between events")
            return False
        return True

    def thigh_mass(self):
        mass = self.parameters["Mass"]
        return (7.3313*mass+3.6883)/1000

    def thigh_com(self, side: str) -> tuple[float, float, float]:
        side = side.capitalize()
        if side not in ["Left", "Right"]:
            raise ValueError("Side must be 'Left' or 'Right'")
        femur_length = self.parameters[f"{side}FemurLength"]
        mass = self.parameters["Mass"]
        return (
            femur_length*(-8.7844332/100000), 
            mass*0.148741316*(-42.118041/100), 
            mass*0.098448042*(2.00427791/100) * -1 if side == "Left" else 1
        )
    
    def thigh_moi(self, side: str) -> tuple[float, float, float]:
        side = side.capitalize()
        if side not in ["Left", "Right"]:
            raise ValueError("Side must be 'Left' or 'Right'")
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
        if side not in ["Left", "Right"]:
            raise ValueError("Side must be 'Left' or 'Right'")
        tibia_length = self.parameters[f"{side}TibiaLength"]
        mass = self.parameters["Mass"]
        return (
            (mass)*0.09004923*(-2.43352222/100), 
            tibia_length*(67.363643/100000), 
            (mass*0.07731125)*(1.71207065/100) * -1 if side == "Left" else 1
        )
        
    def shank_moi(self, side: str) -> tuple[float, float, float]:
        side = side.capitalize()
        if side not in ["Left", "Right"]:
            raise ValueError("Side must be 'Left' or 'Right'")
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
        if side not in ["Left", "Right"]:
            raise ValueError("Side must be 'Left' or 'Right'")
        foot_length = self.parameters[f"{side}FootLength"]
        mass = self.parameters["Mass"]
        return (
            (mass*0.04627387)*(-4.294993/100),
            foot_length*(-42.78009/100000),
            (mass*0.07246637)*(0.6265934/100) * -1 if side == "Left" else 1
        ) # TODO: Still need to check the weird thing Brody does with this in the old code
        
    def foot_moi(self, side: str) -> tuple[float, float, float]:
        side = side.capitalize()
        if side not in ["Left", "Right"]:
            raise ValueError("Side must be 'Left' or 'Right'")
        foot_length = self.parameters[f"{side}FootLength"]
        mass = self.parameters["Mass"]
        return (
            (0.000384786)*(mass)*(foot_length/1000)**2,
            (0.0000518802)*(mass)*(foot_length/1000)**2,
            (0.000364591)*(mass)*(foot_length/1000)**2
        )

    # TODO: Check paths
    def create_scaled_model(self, 
                            unscaled_model_path:str, 
                            marker_set_path:str, 
                            output_dir:str = '.', 
                            ):
        if self.trial_type != "Static" or not self.valid_static():
            raise ValueError("Trial is not a valid static trial")
        
        import os
        os.chdir(output_dir)
        
        import opensim as osim
        scale_tool = osim.ScaleTool()
        scale_tool.setName(self.name)
        
        model_scaler : osim.ModelScaler = scale_tool.getModelScaler()
        model_scaler.setApply(True)
        scaled_model_name = f"{self.name}_scaled.osim"
        model_scaler.setOutputModelFileName(scaled_model_name)
        model_scaler.setOutputScaleFileName(f"{self.name}_scale.xml")
        model_scaler.setMarkerFileName(f"{self.name}.trc")
        
        time_range = osim.ArrayDouble()
        first_time = self.points.time_from_frame(self.points.first_frame)
        last_time = self.points.time_from_frame(self.points.last_frame)
        time_range.set(0, first_time)
        time_range.set(1, last_time)
        model_scaler.setTimeRange(time_range)

        scale_tool.setSubjectMass(self.parameters["Mass"])
        
        # Manual scaling factors
        for side in ["Left", "Right"]:
            for body_part in ["Femur", "Tibia"]:
                side_short = side[0].lower()
                part_length = self.parameters[f"{side}{body_part}Length"]
                base_length = self.base_femur_length if body_part == "Femur" else self.base_tibia_length
                scale_factor = part_length / base_length
                model_scaler.getScaleSet().get(f"{body_part.lower()}_{side_short}").setScaleFactors(osim.Vec3(scale_factor))

        marker_placer : osim.MarkerPlacer = scale_tool.getMarkerPlacer()
        marker_placer.setApply(True)
        marker_model_name = f"{self.name}_marker.osim"
        marker_placer.setOutputModelFileName(marker_model_name)
        marker_placer.setMarkerFileName(f"{self.name}.trc")
        marker_placer.setTimeRange(time_range)
        
        generic_model_maker : osim.GenericModelMaker = scale_tool.getGenericModelMaker()
        generic_model_maker.setModelFileName(unscaled_model_path)
        generic_model_maker.setMarkerSetFileName(marker_set_path)

        scale_tool.printToXML(f"{self.name}_scale_setup.xml")
        scale_tool = osim.ScaleTool(f"{self.name}_scale_setup.xml")
        
        scale_tool.run()
        
        scaled_model = osim.Model(scaled_model_name)
        scaled_model.setName(scaled_model_name.replace(".osim", ""))

        marker_model = osim.Model(marker_model_name)
        marker_model.setName(marker_model_name.replace(".osim", ""))
        
        for model in [scaled_model, marker_model]:
            for side in ["Left", "Right"]:
                side_short = side[0].lower()
                model_body_set: osim.BodySet = model.getBodySet()
                
                thigh: osim.Body = model_body_set.get(f"femur_{side_short}")
                thigh.set_mass(self.thigh_mass())
                thigh.set_mass_center(osim.Vec3(*self.thigh_com(side)))
                thigh.set_inertia(osim.Inertia(*self.thigh_moi(side), 0, 0, 0))

                shank: osim.Body = model_body_set.get(f"tibia_{side_short}")
                shank.set_mass(self.shank_mass())
                shank.set_mass_center(osim.Vec3(*self.shank_com(side)))
                shank.set_inertia(osim.Inertia(*self.shank_moi(side), 0, 0, 0))

                foot: osim.Body = model_body_set.get(f"foot_{side_short}")
                foot.set_mass(self.foot_mass())
                foot.set_mass_center(osim.Vec3(*self.foot_com(side)))
                foot.set_inertia(osim.Inertia(*self.foot_moi(side), 0, 0, 0))
            model.printToXML(model.getName() + ".osim")


    def inverse_kinematics(self, model_path: str, output_path: str | None = None):
        import opensim as osim
        
        if output_path is None:
            logger.warning("No output path specified for inverse kinematics, using working directory")
            output_path = f"{self.name}_ik.mot"
        
        ik_tool = osim.InverseKinematicsTool()
        ik_tool.setName(self.name)
        ik_tool.setModel(model_path)

    def inverse_dynamics(self, model_path:str, output_path: str | None = None):
        import opensim as osim

        if output_path is None:
            logger.warning("No output path specified for inverse dynamics, using working directory")
            output_path = f"{self.name}_id.mot"

        id_tool = osim.InverseDynamicsTool()
        id_tool.setName(self.name)
        id_tool.setModel(model_path)