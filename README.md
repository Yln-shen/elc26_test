source .venv/bin/activate

# 进入项目目录
cd ~/elc26_test

# 激活虚拟环境
woxaindioasw

# 确认激活成功（提示符前应该显示 (elc26-test)）
# 如果没显示，可以手动检查
which python  # 应该显示 /home/radxa/elc26_test/.venv/bin/python

python src/vision/main.py


# 回退到上一个版本（dabe0d9 的上一个）
git reset --hard HEAD~1