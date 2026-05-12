#!/usr/bin/env python3
"""
修复 env_core.py 中的 _encode_obs 函数的 play history 写入逻辑
"""
import re
import os

project_root = os.path.dirname(os.path.dirname(__file__))
src_file = os.path.join(project_root, 'src', 'env_core.py')

with open(src_file, 'r') as f:
    content = f.read()

# 替换注释和检查逻辑
old_code = """        # ---- 704-756: Play history ----
        if hasattr(self, '_play_history'):
            for i, (pl, act) in enumerate(self._play_history):
                if i >= 53:
                    break
                obs[OFS_PLAY + i] = float(act)"""

new_code = """        # ---- 728-756: Play history ----
        if hasattr(self, '_play_history'):
            for i, (pl, act) in enumerate(self._play_history):
                if i >= PLAY_DIM:
                    break
                obs[OFS_PLAY + i] = float(act)"""

content = content.replace(old_code, new_code)

with open(src_file, 'w') as f:
    f.write(content)

print("✅ 修复完成！")
