# Third-party models

`mujoco_menagerie/` is a sparse, shallow checkout of the official
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
Only `franka_emika_panda/` and repository-level documentation are checked out.

Pinned initial revision: `da76818e269b82289eba39808e2fb91d679d6994`.

The files under that checkout remain upstream-owned and unmodified. Project
geometry is added at load time from `src/robot_drawing_repair/scene.py`.

