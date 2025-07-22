import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional

import cv2
import numpy as np
import tyro
from loguru import logger


from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_detector import SingleHandDetector



import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class RetargetingPublisher(Node):
    def __init__(self):
        super().__init__('retargeting_publisher')
        self.publisher_ = self.create_publisher(JointState, 'cmd_allegro', 10)
        self.get_logger().info("Publisher initialized for 'cmd_allegro'.")

    def publish_qpos(self, qpos_cmd):
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = [f'joint_{i}' for i in range(len(qpos_cmd))]
        joint_state.position = qpos_cmd.tolist()
        self.publisher_.publish(joint_state)
        self.get_logger().info(f"Published: {joint_state.position}")

def start_retargeting(queue: multiprocessing.Queue, robot_dir: str, config_path: str):
    rclpy.init()
    retargeting_publisher = RetargetingPublisher()
   
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    hand_type = "Right" if "right" in config_path.lower() else "Left"
    detector = SingleHandDetector(hand_type=hand_type, selfie=False)


    # Different robot loader may have different orders for joints
    # sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    # retargeting_joint_names = retargeting.joint_names
    # retargeting_to_sapien = np.array([retargeting_joint_names.index(name) for name in sapien_joint_names]).astype(int)

    while rclpy.ok():
        start_t = time.time()
        try:
            rgb = queue.get(timeout=50)
        except Empty:
            logger.error(f"Fail to fetch image from camera in 50 secs. Please check your web camera device.")
            return

        _, joint_pos, _, _ = detector.detect(rgb)
        if joint_pos is None:
            logger.warning(f"{hand_type} hand is not detected.")
        else:
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices
            if retargeting_type == "POSITION":
                indices = indices
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            qpos = retargeting.retarget(ref_value)
            # print("qpos: " + ", ".join(f"{pos:.4f}" for pos in qpos))

            qpos_cmd = np.zeros(16)
            # current_pos = leaphand.read_pos()
            # diff = (qpos[0]- current_pos[0] + 3.14) 
            # if  abs(diff)> 0.001:
            #     diff = np.sign(diff) * 0.001
                 
            # qpos_cmd[0] =  current_pos[0] +  diff 

            # qpos_cmd[0] = qpos[0]
            # qpos_cmd[1] = qpos[1]
            # qpos_cmd[2] = qpos[2]
            # qpos_cmd[3] = qpos[3]

            # qpos_cmd[4] = qpos[8] # thumb - middle
            # qpos_cmd[5] = qpos[9]
            # qpos_cmd[6] = qpos[10]
            # qpos_cmd[7] = qpos[11]

            # qpos_cmd[8] = qpos[12] # none
            # qpos_cmd[9] = qpos[13]
            # qpos_cmd[10] = qpos[14]
            # qpos_cmd[11] = qpos[15]

            # qpos_cmd[12] = qpos[4] # thumb - middle 
            # qpos_cmd[13] = qpos[5]
            # qpos_cmd[14] = qpos[6]
            # qpos_cmd[15] = qpos[7]

            # ['1', '0', '2', '3', '12', '13', '14', '15', '5', '4', '6', '7', '9', '8', '10', '11']

            qpos_cmd[0] = qpos[1]
            qpos_cmd[1] = qpos[0]
            qpos_cmd[2] = qpos[2]
            qpos_cmd[3] = qpos[3]

            qpos_cmd[4] = qpos[9] # thumb - middle
            qpos_cmd[5] = qpos[8]
            qpos_cmd[6] = qpos[10]
            qpos_cmd[7] = qpos[11]

            qpos_cmd[8] = qpos[13] # none
            qpos_cmd[9] = qpos[12]
            qpos_cmd[10] = qpos[14]
            qpos_cmd[11] = qpos[15]

            qpos_cmd[12] = qpos[4] # thumb - middle 
            qpos_cmd[13] = qpos[5]
            qpos_cmd[14] = qpos[6]
            qpos_cmd[15] = qpos[7]

            # qpos_cmd[8] = qpos[8]        

            # qpos_cmd = qpos
            print(f"{qpos_cmd[1]:.4f}")
            # print("qpos_cmd: " + ", ".join(f"{pos:.4f}" for pos in qpos_cmd))
            end_t = time.time()
            # print(f"time: {end_t - start_t:.4f} s")

            # leaphand.set_allegro(qpos_cmd)
            # Publish qpos_cmd
            retargeting_publisher.publish_qpos(qpos_cmd)

            # print("Position: " + str(leaphand.read_pos()))
            # time.sleep(0.02)
            # a = input("test")

    rclpy.shutdown()




def produce_frame(queue: multiprocessing.Queue, camera_path: Optional[str] = None):
    if camera_path is None:
        # print("test")
        cap = cv2.VideoCapture('/dev/video22')
    else:
        # print("test2")
        cap = cv2.VideoCapture(camera_path)

    print("cap states: ", cap.isOpened())

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("waiting for camera")
            continue
        frame = image
        print("frame shape: ", frame.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        queue.put(image)
        # time.sleep(1 / 60.0)
        # cv2.imshow("demo", frame)
        
        cv2.imshow("demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main(
    robot_name: RobotName, retargeting_type: RetargetingType, hand_type: HandType, camera_path: Optional[str] = None
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
        camera_path: the device path to feed to opencv to open the web camera. It will use 0 by default.
    """
    # config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    # robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    
    # Update these paths to your local paths
    config_path = Path("../../dex_retargeting/configs/teleop/leap_hand_right_dexpilot.yml")
    robot_dir = Path("../../assets/robots/hands")

    print("Start retargeting with config {config_path}")

    queue = multiprocessing.Queue(maxsize=1)
    producer_process = multiprocessing.Process(target=produce_frame, args=(queue, camera_path))
    # print("test3")

    # print("test4")
    consumer_process = multiprocessing.Process(target=start_retargeting, args=(queue, str(robot_dir), str(config_path)))

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()
    time.sleep(5)

    print("done")


if __name__ == "__main__":
    tyro.cli(main)
