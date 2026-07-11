**Real-Time Face Recognition(OpenCV + LBPH)

A real-time face recognition system using OpenCV's Haar Cascade classifiers 
for face and eye detection, combined with the LBPH (Local Binary Patterns 
Histograms) algorithm for identity recognition through a live webcam feed.

**Tech Stack
- Python
- OpenCV (opencv-contrib-python)
- NumPy
- Pillow

**Execution
1. `face train.py` scans labeled face images inside the `images/` folder, 
   detects faces using a Haar Cascade classifier, and trains an LBPH 
   recognizer on them. This produces two files: `trainner.yml` (the trained 
   model) and `labels.pickle` (a mapping of names to numeric IDs).
2. `Face Recognition.py` opens the webcam, detects faces in real time, and 
   labels each detected face with the person's name if the model recognizes 
   them with sufficient confidence.

**Setup
1. Install dependencies:
   pip install -r requirements.txt
2. Add face images inside `images/<person_name>/` — one folder per person, 
   8-15 images each.
3. Train the model:
   python "face train.py"
4. Run recognition:
   python "Face Recognition.py"
5. Press 'q' to quit the webcam window.
