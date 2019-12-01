import os
import torille
import random

from torille import ToribashControl

"""
 Test to make sure that replays were loaded into the correct location
 and can be played. Will list the number of replays and also play 
 a random replay found in the folder. To play specific replays see
 utils.viz_tools.visuals.watch_replay
"""

if __name__ == "__main__":
    print("Checking for replays")
    path = os.path.dirname(torille.__file__)
    replays_path = os.path.join(path, 'toribash/replay')
    replays = [f for f in os.listdir(replays_path) if '.rpl' in f]
    if len(replays) == 0:
        raise ValueError("There does not seem to be any replays at {} \n Make sure to download replays from data/ or from {}".format(
    replays_path, 'https://forum.toribash.com/forumdisplay.php?f=10.'
    ))
    else:
        print("Found {} replays".format(len(replays)))
        print("Playing random replay")
        sample = random.sample(replays, 1)[0]
        try:
            controller = ToribashControl(draw_game=True)
            controller.init()
            controller.finish_game()
            states = controller.read_replay(sample)
            controller.close()
        except RuntimeError as inst:
            print("Manual controller shutdown. This could be caused by a variety " + 
            "of issues. Please check torille install.")
            controller.close()
            print("error: {}".format(inst))

