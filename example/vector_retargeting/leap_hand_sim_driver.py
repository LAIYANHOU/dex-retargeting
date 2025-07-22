#!/usr/bin/env python3
import sys
import time
from typing import Optional

import cv2
import numpy as np
import pygame
from pygame.locals import KEYDOWN

from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_detector import SingleHandDetector


def run_sim_teleop(
            config_path=str(config_path),
            robot_dir=str(robot_dir),
            hand_type=hand_type,
            camera_path=camera_path,
            keyboard=keyboard,
    ):
    """
    在 Isaac Sim 中使用 Teleop (鍵盤或 MediaPipe) 驅動 LeapHand。  
    Args:
        config_path: retargeting config 的檔案路徑
        camera_path: 可選的攝影機路徑
        keyboard: 如果為 True，使用鍵盤輸入；否則使用 MediaPipe
        gui: 如果為 True，開啟 GUI；否則 headless 模式
    """
    # 1. 啟動 SimulationApp（必須最先）
    from omni.isaac.kit import SimulationApp
    headless = not gui
    app = SimulationApp({
        "headless": headless,
        "extensions": {"disable": ["omni.kit.test"]},
    })

    # 2. SimulationApp 建立後，再 import 需要的模組
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction

    # 3. 載入 USD
    usd_path = "/home/lai/ros2_ws/src/franka_description/robots/fr3/fr3_with_leaphand/fr3_with_leaphand.usd"
    prim_path = "/World/Franka"
    world = World()
    add_reference_to_stage(usd_path, prim_path)
    world.reset()

    # 4. 建立 Articulation
    robot = SingleArticulation(prim_path)
    robot.initialize()

    # 5. Retargeting 設定
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    hand_type = "Right" if "right" in config_path.lower() else "Left"
    detector = SingleHandDetector(hand_type=hand_type, selfie=False)
    
    # 6. 找出手部 DOF indices
    dof_names = robot.dof_names
    hand_idx = [i for i, n in enumerate(dof_names) if not n.startswith("fr3_")]
    driver = LeapHandSimDriver(robot, hand_idx, step=0.05)

    # 7. 如果使用 MediaPipe，就打開攝影機
    cap = None
    if not keyboard:
        cap = cv2.VideoCapture(camera_path or 0)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {camera_path}")

    # 8. 初始化 pygame
    pygame.init()
    screen = pygame.display.set_mode((300, 300))
    pygame.display.set_caption("LeapHand Sim Teleop")

    try:
        while app.is_running():
            # 處理鍵盤輸入
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    key = pygame.key.name(event.key)
                    driver.handle_key_event(key)

            # 如果使用 MediaPipe，讀取影像並 retarget
            if not keyboard and cap:
                ret, frame = cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    _, joint_pos, _, _ = detector.detect(rgb)
                    if joint_pos is not None:
                        indices = retargeting.optimizer.target_link_human_indices
                        if retargeting.optimizer.retargeting_type == "POSITION":
                            ref = joint_pos[indices, :]
                        else:
                            org, task = indices
                            ref = joint_pos[task, :] - joint_pos[org, :]
                        qpos = retargeting.retarget(ref)
                        # 重新排序到 q
                        q = np.zeros_like(qpos)
                        q[0], q[1], q[2], q[3] = qpos[1], qpos[0], qpos[2], qpos[3]
                        q[4], q[5], q[6], q[7] = qpos[9], qpos[8], qpos[10], qpos[11]
                        q[8], q[9], q[10], q[11] = qpos[13], qpos[12], qpos[14], qpos[15]
                        q[12], q[13], q[14], q[15] = qpos[4], qpos[5], qpos[6], qpos[7]
                        driver.update(q)

            # 應用目前角度到模擬
            driver.apply()
            world.step(render=True)
    finally:
        if cap:
            cap.release()
        pygame.quit()
        app.close()


class LeapHandSimDriver:
    """控制 Isaac Sim 中的 LeapHand 關節"""

    def __init__(self, robot, hand_idx: list, step: float = 0.05):
        self.robot = robot
        self.hand_idx = hand_idx
        self.joint_angles = np.zeros(len(hand_idx), dtype=float)
        self.step = step

    def update(self, joint_angles: np.ndarray):
        current = self.robot.get_joint_positions()
        current[self.hand_idx] = joint_angles
        self.robot.apply_action(ArticulationAction(joint_positions=current))

    def apply(self):
        self.update(self.joint_angles)

    def handle_key_event(self, key_name: str):
        key_map = {
            'q': (0, +1), 'a': (0, -1),
            'w': (1, +1), 's': (1, -1),
            'e': (2, +1), 'd': (2, -1),
            'r': (3, +1), 'f': (3, -1),
            't': (4, +1), 'g': (4, -1),
            'y': (5, +1), 'h': (5, -1),
            'u': (6, +1), 'j': (6, -1),
            'i': (7, +1), 'k': (7, -1),
        }
        if key_name in key_map:
            idx, direction = key_map[key_name]
            self.joint_angles[idx] += direction * self.step
            self.joint_angles[idx] = float(np.clip(self.joint_angles[idx], 0.0, 1.5))
