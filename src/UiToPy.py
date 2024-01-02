import os
os.system(command="pyuic5 -o colormetric.py colormetric.ui")

os.system(command="pyrcc5 ..\ico\ico.qrc -o ico_rc.py")