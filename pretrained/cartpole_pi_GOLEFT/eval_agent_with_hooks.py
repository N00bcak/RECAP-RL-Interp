import argparse
import gin
import os
import util
import numpy as np
import time
import torch
from pathlib import Path
import matplotlib as mpl
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageSequenceClip

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log-dir', help='Directory of logs.')
    parser.add_argument(
        '--model-filename', default='model.npz',
        help='File name of the model to evaluate.')
    parser.add_argument(
        '--n-episodes', help='Number of episodes to evaluate.',
        type=int, default=3)
    parser.add_argument(
        '--seed', help='Random seed for evaluation.', type=int, default=1)
    parser.add_argument(
        '--prefix', default='default',
        help = 'Prefix for logging and video names.'
    )
    parser.add_argument(
        '--pos', type=int, default=2,
        help='Position to ablate in the attention matrix.'
    )
    parser.add_argument(
        '--dim', type=int, default=0,
        help='Dimension to ablate in the attention matrix.'
    )
    config, _ = parser.parse_known_args()
    return config

def main(config):

    EXPT = Path(config.log_dir, config.prefix)
    EXPT.mkdir(parents=True, exist_ok=True)

    logger = util.create_logger(name=f'test_log', log_dir=config.log_dir)
    task = util.create_task(logger=logger)
    task.seed(config.seed)

    def get_permutation_ix():
        """
        Get the permutation index for the task.
        """
        # This is a hack to get the permutation index from the task.
        # The task should have a method to get the permutation index.
        return task.perm_ix

    def mask(pos, dim=1):
        """
        Mask out the activations at that specific position.
        """
        def ablation(x):
            # Mask out the activations at the specified position in the tensor x
            if dim == 1:
                newp = get_permutation_ix().tolist().index(pos)
            else:
                newp = pos
            x.index_fill_(dim=dim, index=torch.tensor([newp]), value=-1)
            # print(x)
            return x

        return ablation
    
    def set_to(pos, dim = 1, to_val = 1):
        """
        Set the activations at that specific position to a value.
        """
        def ablation(x):
            # Set the activations at the specified position in the tensor x
            if dim == 1:
                newp = get_permutation_ix().tolist().index(pos)
            else:
                newp = pos
            x.index_fill_(dim=dim, index=torch.tensor([newp]), value = to_val)
            return x

        return ablation

    def zero_lstm_state():
        """
        Zero out the activations.
        """
        def ablation(x):
            x[0].zero_()
            x[1].zero_()
            return x

        return ablation
    
    def mean_ablate(lst):
        """
        Ablate the activations in the tensor x by taking the mean of the specified entries.
        The first entry should be the new value, and the
        second entry should be the coordinates to modify.
        """
        def ablation(x):
            for i, coords in enumerate(lst):
                # print(f'Modifying {i}th entry with value {new_value} at coords {coords}')
                coords = (coords[0], get_permutation_ix().tolist().index(coords[1]))
                x[coords] = x.mean()
            return x

        return ablation

    def modify(lst):
        """
        Modify the activations in the tensor x.
        The first entry should be the new value, and the
        second entry should be the coordinates to modify.
        """
        def ablation(x):
            for i, (new_value, coords) in enumerate(lst):
                # print(f'Modifying {i}th entry with value {new_value} at coords {coords}')
                coords = (coords[0], get_permutation_ix().tolist().index(coords[1]))
                x[coords] = new_value
            return x

        return ablation

    def compose(ablations):
        """
        Compose multiple ablation functions.
        """
        def ablation(x):
            for fn in reversed(ablations):
                x = fn(x)
            return x

        return ablation

    hook_fns = {
        'pi_layer_activations': {
            # There seems to be a pretty strange pattern in the attention matrix.
            # There are distinct activation patterns for when the pole is tilted slightly to the left or right.
            # Since this pattern induces the model to move in the same direction of the tilt so as to
            # correct it, we call this the "NOPE" pattern.
            'attention_matrix': compose([
                modify([
                    (1., (6, 3)), (1., (6, 0)), (1., (6, 1)),
                    (1., (15, 4)), (1., (15, 0)), (1., (15, 2)),
                ]),
                set_to(0, 0, 1), set_to(1, 0, 1), set_to(2, 0, 1),
                set_to(6, 0, 1), set_to(7, 0, 1), set_to(8, 0, 1), 
                set_to(13, 0, 1), set_to(14, 0, 1),
                mask(3, 0), mask(4, 0), mask(5, 0), 
                mask(9, 0), mask(10, 0), mask(11, 0), 
                mask(12, 0), mask(15, 0)
            ])
            
            # 'attention_matrix': compose(
            # [
            #     modify([
            #         (1., (0, 1)), (1., (6, 1)), (1., (12, 1)), (1., (13, 1)), (1., (14, 2)), (1., (14, 3)),
            #         # *[(1, (i, 1)) for i in range(15)],
            #         *[(1, (i, 4)) for i in range(15)]
            #     ]),
            # ]
            # )
        },
    }


    solution = util.create_solution(device='cpu:0', hook_fns = hook_fns)
    model_file = os.path.join(config.log_dir, config.model_filename)
    solution.load(model_file)

    print(f'Loaded model from {model_file}')
    print(solution.pi_layer)
    print(solution.net)

    for name, module in solution.pi_layer.named_modules():
        print(name, module)

    ATTENTION_CMAP = mpl.colormaps.get_cmap('viridis')
    def attention_to_rgb(attention_matrix):
        """
        Convert attention matrix to RGB image.
        Expected shape: (batch, d_feats, d_obspos)
        """
        attention_matrix = attention_matrix.squeeze()
        rgb_image = ATTENTION_CMAP(attention_matrix)[..., :3]  # Drop alpha channel
        return (rgb_image * 255).astype(np.uint8)

    VIDEOS = Path(EXPT, 'videos')
    VIDEOS.mkdir(parents=True, exist_ok=True)


    HOOK_DATA = Path(EXPT, 'hook_data')
    HOOK_DATA.mkdir(parents=True, exist_ok=True)
    def make_video(frames, filename):
        """
        Save frames as a video.
        """

        clip = ImageSequenceClip(list(frames), fps=24)
        clip.write_videofile(str(filename), codec='libx264', audio=False)

    rewards = []
    time_costs = []
    for ep in range(config.n_episodes):
        start_time = time.perf_counter()
        reward, frames, permutations = task.rollout(solution=solution, evaluation=True)

        # Shape: (n_frames, d_feats, d_obspos)
        attention_matrices = np.array([
            x['pi_layer_activations']['attention_matrix'].numpy() 
            for x in solution.full_hook_data
        ])

        # Scale the matrix to the desired size.
        attention_matrices = np.kron(attention_matrices, np.ones((1, 16, 16)))

        # Shape: (n_frames, h, w, c)
        frames = np.array(frames[:-1])
        # Attach the frames to the hook data.
        for i in range(len(solution.full_hook_data)):
            solution.full_hook_data[i]['permutation'] = permutations[i]
        # for i, frame in enumerate(frames):
            # solution.full_hook_data[i]['permutation'] = task.perm_ix
            # solution.full_hook_data[i]['video_frame'] = frame

        # print(attention_matrices.shape, frames.shape)
        # Now shape: (n_frames, d_feats, d_obspos, 3)
        # attention_matrices = attention_to_rgb(attention_matrices)
        # # print("RENDERED SHAPE")
        # # print(attention_matrices.shape, frames.shape)

        # pad_width = (frames.shape[1] - attention_matrices.shape[1])
        # pad_height = (frames.shape[2] // 2 - attention_matrices.shape[2])
        # # Now hopefully, shape: (n_frames, h, w, 3)
        # attention_matrices = np.pad(
        #     attention_matrices,
        #     ((0, 0), (pad_width, 0), (pad_height // 2, pad_height // 2), (0, 0)),
        #     mode='constant', constant_values = 255
        # )
        # print("POST-PADDING SHAPE")
        # print(attention_matrices.shape, frames.shape)

        # def add_text(frame, text, position = (10, 10), font_size=20):
        #     """
        #     Add text to a frame.
        #     """
        #     img = Image.fromarray(frame.copy())
        #     draw = ImageDraw.Draw(img)
        #     font = ImageFont.load_default(size = font_size)
        #     draw.text(position, text, fill=(0, 0, 0), font=font)
        #     return np.array(img)

        # obs = [[round(x[0], 4) for x in solution.full_hook_data[i]["pi_layer_activations"]["obs"].tolist()] for i in range(len(solution.full_hook_data))]
        # print(obs)
        # Add the observation space to the frames.
        # frames = np.array([
        #     add_text(frame, f'x,v,t,w\n{obs[i]}', position = (20, 20))
        #     for i, frame in enumerate(frames)
        # ])

        # frames = np.concatenate([frames, attention_matrices], axis=-2)
        # Save as video.
        make_video(frames, VIDEOS / f'video_ep_{ep}.mp4')
        # Save the hook data.
        torch.save(solution.full_hook_data, 
            HOOK_DATA / f'hook_data_ep_{ep}.pt'
        )
        
        time_cost = time.perf_counter() - start_time
        rewards.append(reward)
        time_costs.append(time_cost)
        logger.info(f'Episode: {ep + 1}, reward: {reward:.2f}')

    logger.info(f'Avg reward: {np.mean(rewards):.2f}, sd of mean: {np.std(rewards) / np.sqrt(len(rewards)):.2f}')
    logger.info(f'Time per rollout: {np.mean(time_costs)}s')


if __name__ == '__main__':
    args = parse_args()
    gin.parse_config_file(os.path.join(args.log_dir, 'config.gin'))
    main(args)
