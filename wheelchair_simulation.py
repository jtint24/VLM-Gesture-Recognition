import pybullet as p
import pybullet_data
import time

class wheelchair_sim:
    def __init__(self, gesture_semantics, recognizer, time_limit_seconds=60):
        self.gesture_semantics = gesture_semantics
        self.recognizer = recognizer
        self.time_limit_seconds = time_limit_seconds
        self.start_time_seconds = None
        self.robot_id = None
        self.left_wheel = None
        self.right_wheel = None
        self.caster_wheels = []

    def setup_simulation(self):
        # Connect to physics server
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load environment
        p.loadURDF("plane.urdf")
        
        # Load wheelchair URDF with correct parameters
        start_pos = [0, 0, 0.15]  # Adjusted for base height (0.1m) + clearance
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF("wheel_model.urdf", start_pos, start_orientation)

        # Configure joints from your URDF structure
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode("utf-8")
            
            if joint_name == "wheelbase_leftwheel":
                self.left_wheel = i
                p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL, force=500)
            elif joint_name == "wheelbase_rightwheel":
                self.right_wheel = i
                p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL, force=500)
            elif "caster" in joint_name.lower():  # Detect caster joints
                self.caster_wheels.append(i)
                # Allow free rotation for caster wheels
                p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL, force=0)

        if None in [self.left_wheel, self.right_wheel]:
            raise ValueError("Drive wheel joints not found in URDF")

        print(f"Drive wheels: Left[{self.left_wheel}], Right[{self.right_wheel}]")
        print(f"Caster wheels: {self.caster_wheels}")

    def move(self, linear, angular):
        """Differential drive control based on URDF parameters"""
        wheel_radius = 0.09  # From URDF cylinder radius
        wheel_separation = 0.5  # 0.25m x2 from joint positions
        
        left_speed = (linear - angular * wheel_separation/2) / wheel_radius
        right_speed = (linear + angular * wheel_separation/2) / wheel_radius
        
        p.setJointMotorControl2(self.robot_id, self.left_wheel,
                              p.VELOCITY_CONTROL, 
                              targetVelocity=left_speed,
                              force=500)
        p.setJointMotorControl2(self.robot_id, self.right_wheel,
                              p.VELOCITY_CONTROL,
                              targetVelocity=right_speed,
                              force=500)

    def interpret(self, image):
        recognized = self.recognizer.recognize(image, list(self.gesture_semantics.keys()))
        
        # Ensure we use the first recognized gesture as a string
        if recognized and isinstance(recognized, list):
            return self.gesture_semantics.get(recognized[0], None)
        elif isinstance(recognized, str):
            return self.gesture_semantics.get(recognized, None)
        return None

    def update(self, now, image):
        if (now - self.start_time_seconds) > self.time_limit_seconds:
            return False

        command = self.interpret(image)
        if command:
            print(f"Gesture detected: {command}")
            if command == "forward":
                self.move(linear=2, angular=0)
            elif command == "backward":
                self.move(linear=-1, angular=0)
            elif command == "left":
                self.move(linear=0, angular=1)
            elif command == "right":
                self.move(linear=0, angular=-1)
            elif command == "stop":
                self.move(0, 0)
        else:
            self.move(0, 0)  # Stop when no gesture detected
            
        p.stepSimulation()
        time.sleep(1./240.)
        return None

    def start(self):
        self.start_time_seconds = time.time()
        self.setup_simulation()