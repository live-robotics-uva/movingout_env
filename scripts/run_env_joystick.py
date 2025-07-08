import argparse
import pygame
from moving_out.benchmarks.moving_out import MovingOutEnv
from moving_out.utils.joystick_controller import JoystickController

def test_env(map_name):
    env = MovingOutEnv(use_state=False, map_name=map_name)

    joystick_controller = JoystickController()

    fps = 10
    clock = pygame.time.Clock()

    while True:
        clock.tick(fps)
        print(f"FPS: {clock.get_fps()}")
        env.render("human")
        action = joystick_controller.get_actions()
        # print(action)
        obs, rew, terminated, truncated, info = env.step(action)

        if terminated:
            env.reset(map_name=map_name)
        


if __name__ == "__main__":
    procedure = """
    ===================================================
            Experiment Participation Procedure
    ===================================================
    
    Welcome to participate in our experiment. This 
    experiment aims to study the behavior when people 
    play the game 'Moving Out' with an AI agent.
    
    Operation Instructions:
    You need to use a joystick to control the robot's 
    movement and item handling. 
    Use the joystick to control the direction of the 
    robot's movement and hold the R button to grab or 
    release items.
    
    After the study, you will receive a questionnaire. 
    Please answer the questions based on your experience.
    
    Thank you!
    ===================================================
    """
    parser = argparse.ArgumentParser(description="Process JSON file and ID.")
    parser.add_argument(
        "--map_name", type=str, default="HandOff", help="The ID number to extract"
    )
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])

    test_env(args.map_name)
