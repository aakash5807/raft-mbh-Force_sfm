Chain Snatching Detection using Force-SfM (RAFT + MBH)

This project implements an advanced Chain Snatching Detection System using:

RAFT Optical Flow

MBH (Motion Boundary Histogram)

Force-based Structure from Motion (Force-SfM)

Instead of detecting simple contact, this system estimates physical force of interaction using motion-derived acceleration.

🚀 Core Idea

Normal interaction ≠ Snatching.

Snatching involves:

Sudden hand acceleration

Neck displacement

Reaction delay

Short violent motion spike

We compute a Force Index from motion dynamics to detect aggressive pulling events.

🧠 Technologies Used

Python

PyTorch

RAFT Optical Flow

MediaPipe Pose

OpenCV

NumPy

Matplotlib

git clone https://github.com/aakash5807/raft-mbh-Force_sfm.git
cd raft-mbh-Force_sfm
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
