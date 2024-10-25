import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding


class Gazebo_Linefollow_Env(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src/linefollow_ros/launch/linefollow_world.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        self.action_space = spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected
        self.framecount = 0


    def process_image(self, data):
        '''
            @brief Coverts data into a opencv image and displays it
            @param data : Image data from ROS

            @retval (state, done)
        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        self.framecount += 1

        # cv2.imshow("raw", cv_image)
    

        NUM_BINS = 3
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        done = False

        # TODO: Analyze the cv_image and compute the state array and
        # episode termination condition.
        #
        # The state array is a list of 10 elements indicating where in the
        # image the line is:
        # i.e.
        #    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] indicates line is on the left
        #    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] indicates line is in the center
        #
        # The episode termination condition should be triggered when the line
        # is not detected for more than 30 frames. In this case set the done
        # variable to True.
        #
        # You can use the self.timeout variable to keep track of which frames
        # have no line detected.

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        #thresholding
        _, binary_image = cv2.threshold(gray_image, 130, 255, cv2.THRESH_BINARY_INV)

        #get sample row
        height, width = binary_image.shape
        line_section = binary_image[int(height * 0.8), :]
        interval_length = int(width/10)


        white_pixel_indices = np.where(line_section == 255)[0]

        if len(white_pixel_indices) > 0:
            center_index = int(np.mean(white_pixel_indices))
            # print(f"{center_index/width}") 
            # print(f" width: {width}")
            line_detected = True
        else:
            center_index = -1 
            line_detected = False

        if line_detected:
            max_index = int(float(center_index/width)*10)
            state[max_index] = 1
            self.timeout = 0
        else:
            self.timeout += 1

        if self.timeout > 30:
            done = True
    
        state_text = f"State: {state}"
         # Adjust the position to be within the frame bounds (centered at the top)
        position = (int(width * 0.05), int(height * 0.1))  # 5% from left, 10% from top

        # Put the text on the frame with a color that contrasts the background
        cv2.putText(binary_image, state_text, position, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)  # green text

        cv2.imshow("binary", binary_image)
        cv2.waitKey(1)  # delay to ensure the image is rendered

        # OLD CODE
        # max = 0
        # max_index = -1
        # start_pixel = 0
        # for i in range(0,10):
        #     sum = 0
        #     for j in range(start_pixel,interval_length):
        #         sum += line_section[j]
        #     if sum > max:
        #         max = sum
        #         max_index = i
        #     start_pixel += interval_length

        # if max_index != -1 and max > 1000:
        #     state[max_index] = 1
        #     self.timeout = 0
        # else:
        #     self.timeout += 1

        # if self.timeout > 30:
        #     done = True

        return state, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()

        if action == 0:  # FORWARD
            vel_cmd.linear.x = 0.4
            vel_cmd.angular.z = 0.0
        elif action == 1:  # LEFT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.5
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.5

        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw', Image,
                                              timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.process_image(data)

        reward = -50
        # Set the rewards for your action

        # if not done:
        #     if action == 0:  # FORWARD
        #         if 1 in state:
        #             if state.index(1) == 4:  # Center of the line
        #                 reward = 50
        #             elif state.index(1) in [3, 5]:  # Close to center
        #                 reward = 30
        #             else:
        #                 reward = 10  # Far from center
        #     elif action == 1:  # LEFT
        #         if 1 in state:
        #             reward = 20 if state.index(1) in [0, 1, 2, 3] else 0  # Penalize wrong turns
        #     else: # RIGHT
        #         if 1 in state:
        #             reward = 20 if state.index(1) in [6, 7, 8, 9] else 0
        # else:
        #     reward = -50  # Penalize losing the line


        # if not done:
        #     if action == 0:  # FORWARD
        #         reward = 10
        #     elif action == 1:  # LEFT
        #         if 1 in state:
        #             if state.index(1) in [0, 1, 2]: 
        #                 reward = 60 
        #             elif state.index(1) == 3:
        #                 reward = 40 
        #         else:
        #             reward = 20
        #     else: # RIGHT
        #         if 1 in state:
        #             if state.index(1) in [7, 8, 9]: 
        #                 reward = 60 
        #             elif state.index(1) == 6:
        #                 reward = 40 
        #         else:
        #             reward = 20 
        # else:
        #     reward = -50

        if not done:
            if action == 0:  # FORWARD
                if 1 in state:  
                    if state.index(1) == 4:  # Center of the line
                        reward = 60  # High reward for staying centered and moving forward
                    elif state.index(1) in [3, 5]:  # Near the center
                        reward = 40
                    else:
                        reward = -10  # Further from center but still moving forward
                else:
                    reward = -5  # Small reward for moving forward if no line detected

            elif action == 1:  # LEFT
                if 1 in state:
                    if state.index(1) in [0, 1, 2]:  # Correct turn
                        reward = 55
                    elif state.index(1) in  [3, 4]:  # slightly to the right
                        reward = 40
                    elif state.index(1) in [8,9]: # left turn very bad
                        reward = -15
                    else:
                        reward = -10  # penalize incorrect turn
                else:
                    reward = -3  # no reward for unnecessary turning

            else:  # RIGHT
                if 1 in state:
                    if state.index(1) in [7, 8, 9]:  # Correct turn
                        reward = 45
                    elif state.index(1) == [5,6]:  # slightly to the right
                        reward = 35
                    elif state.index(1) == [0, 1, 2]:
                        reward = -19
                    else:
                        reward = -6  # Penalize incorrect turn
                else:
                    reward = 5  # little reward for unnecessary turning
            if len(self.episode_history) >= 3:
                if (self.episode_history[-1] == 1 and self.episode_history[-2] == 2) or \
                (self.episode_history[-1] == 2 and self.episode_history[-2] == 1):
                    reward = -3  # penalize left-right oscillations

        else:
            reward = -50  # penalize for losing the line


        return state, reward, done, {}

    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # read image data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw',
                                              Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(data)

        return state
