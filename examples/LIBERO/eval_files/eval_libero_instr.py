import dataclasses
import json
import logging
import pathlib

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark

from examples.LIBERO.eval_files.eval_libero import (
    Args,
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    _get_libero_env,
    _quat2axisangle,
)
from examples.LIBERO.eval_files.model2libero_interface import ModelClient


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


def _get_max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220  # longest training demo has 193 steps
    if task_suite_name == "libero_object":
        return 280  # longest training demo has 254 steps
    if task_suite_name == "libero_goal":
        return 300  # longest training demo has 270 steps
    if task_suite_name == "libero_10":
        return 520  # longest training demo has 505 steps
    if task_suite_name == "libero_90":
        return 400  # longest training demo has 373 steps
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _slugify(text: str, max_len: int = 80) -> str:
    slug = "".join(char if char.isalnum() else "_" for char in text.lower())
    slug = "_".join(part for part in slug.split("_") if part)
    return slug[:max_len].strip("_") or "task"


def _success_video_name(task_description: str) -> str:
    sentence = task_description.strip()
    if not sentence:
        return "instr.mp4"
    return f"{sentence[:1].upper()}{sentence[1:]}.mp4"


def eval_libero_instr(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    max_steps = _get_max_steps(args.task_suite_name)

    client_model = ModelClient(
        policy_ckpt_path=args.pretrained_path,
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
        fast_inference=args.fast_inference,
    )

    total_tasks = 0
    total_attempts = 0
    total_successes = 0

    for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        max_attempts = min(args.num_trials_per_task, len(initial_states))
        found_success = False
        success_episode_idx = None

        try:
            for episode_idx in tqdm.tqdm(range(max_attempts), desc=f"Task {task_id}", leave=False):
                logging.info(f"\nTask: {task_description}")
                logging.info(f"Starting episode {episode_idx + 1}...")

                client_model.reset(task_description=task_description)
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])

                t = 0
                step = 0
                done = False
                replay_images = []

                while t < max_steps + args.num_steps_wait:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(
                        obs["robot0_eye_in_hand_image"][::-1, ::-1]
                    )
                    replay_images.append(img)

                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )

                    observation = {
                        "observation.primary": np.expand_dims(img, axis=0),
                        "observation.wrist_image": np.expand_dims(wrist_img, axis=0),
                        "observation.state": np.expand_dims(state, axis=0),
                        "instruction": [str(task_description)],
                    }

                    example_dict = {
                        "image": [
                            observation["observation.primary"][0],
                            observation["observation.wrist_image"][0],
                        ],
                        "lang": observation["instruction"][0],
                    }

                    response = client_model.step(example=example_dict, step=step)
                    raw_action = response["raw_action"]

                    world_vector_delta = np.asarray(
                        raw_action.get("world_vector"), dtype=np.float32
                    ).reshape(-1)
                    rotation_delta = np.asarray(
                        raw_action.get("rotation_delta"), dtype=np.float32
                    ).reshape(-1)
                    open_gripper = np.asarray(
                        raw_action.get("open_gripper"), dtype=np.float32
                    ).reshape(-1)
                    gripper = _binarize_gripper_open(open_gripper)

                    if not (
                        world_vector_delta.size == 3
                        and rotation_delta.size == 3
                        and open_gripper.size == 1
                    ):
                        raise ValueError(
                            f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                            f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                        )

                    delta_action = np.concatenate(
                        [world_vector_delta, rotation_delta, gripper], axis=0
                    )

                    obs, reward, done, info = env.step(delta_action.tolist())
                    if done:
                        found_success = True
                        success_episode_idx = episode_idx
                        total_successes += 1
                        break
                    t += 1
                    step += 1

                total_attempts += 1

                if done:
                    video_path = pathlib.Path(args.video_out_path) / _success_video_name(task_description)
                    imageio.mimwrite(
                        video_path,
                        [np.asarray(frame) for frame in replay_images],
                        fps=10,
                    )
                    logging.info(
                        f"Success video saved to {video_path} (episode {episode_idx + 1})"
                    )
                    break

                logging.info(
                    f"Episode {episode_idx + 1} failed; trying next initial state."
                )

            if not found_success:
                logging.warning(
                    f"No success found for task {task_id:03d}: {task_description} "
                    f"within {max_attempts} attempts."
                )

            logging.info(
                f"Task {task_id:03d} finished: success={found_success}, "
                f"success_episode={success_episode_idx if success_episode_idx is not None else 'N/A'}"
            )
        finally:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()

        total_tasks += 1

    logging.info(f"Total tasks: {total_tasks}")
    logging.info(f"Total attempts: {total_attempts}")
    logging.info(f"Total successful tasks: {total_successes}")


if __name__ == "__main__":
    tyro.cli(eval_libero_instr)