# =========================================================
# MATPLOTLIB (MUST BE AT TOP)
# =========================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================================================
# STANDARD IMPORTS
# =========================================================
from io import BytesIO
from pathlib import Path
from datetime import datetime, date
import os
import sqlite3

import cv2
import joblib
import numpy as np
import pandas as pd

from flask import (
    Flask,
    render_template,
    Response,
    request,
    redirect,
    url_for,
    flash,
    send_file,
)

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
FACES_DIR = BASE_DIR / "faces"  # optional (kept)
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

DB_PATH = BASE_DIR / "attendance.db"
ENCODINGS_PATH = BASE_DIR / "face_encodings.pkl"

MODELS_DIR.mkdir(exist_ok=True)
FACES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# =========================================================
# FLASK APP
# =========================================================
app = Flask(__name__)
app.secret_key = "uel-face-attendance-secret"


# =========================================================
# SQLITE HELPERS
# =========================================================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def table_exists(cur, name: str) -> bool:
    row = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def get_columns(cur, table: str):
    return [r["name"] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]


def ensure_column(cur, table: str, col_name: str, col_sql: str):
    """
    Adds a missing column safely.
    col_sql example: "roll TEXT NOT NULL DEFAULT ''"
    """
    cols = get_columns(cur, table)
    if col_name not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_sql}")


def init_db():
    """
    ✅ FIXES your repeated errors:
    - If DB already exists with old schema, we ADD missing columns.
    - We create indexes ONLY after needed columns exist.
    """
    db = get_db()
    cur = db.cursor()

    # --- Students table ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            roll TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            course TEXT DEFAULT '',
            section TEXT DEFAULT ''
        )
    """)

    # --- Attendance table (create if new) ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll TEXT NOT NULL,
            att_date TEXT NOT NULL,
            att_time TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'auto',
            FOREIGN KEY (roll) REFERENCES students(roll)
        )
    """)

    # --- MIGRATION: ensure columns exist in case DB was created earlier differently ---
    # If your old table was missing roll/mode/time/date, these lines prevent "no such column"
    ensure_column(cur, "attendance", "roll", "roll TEXT")
    ensure_column(cur, "attendance", "att_date", "att_date TEXT")
    ensure_column(cur, "attendance", "att_time", "att_time TEXT")
    ensure_column(cur, "attendance", "mode", "mode TEXT NOT NULL DEFAULT 'auto'")

    # Also ensure student columns exist (rare but safe)
    ensure_column(cur, "students", "course", "course TEXT DEFAULT ''")
    ensure_column(cur, "students", "section", "section TEXT DEFAULT ''")

    # --- Indexes (only after columns exist) ---
    # Unique: one attendance per student per day
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_attendance_unique
        ON attendance(roll, att_date)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_attendance_date
        ON attendance(att_date)
    """)

    db.commit()
    db.close()


init_db()


# =========================================================
# MODELS (OpenCV DNN)
# =========================================================
FACE_PROTO = MODELS_DIR / "deploy.prototxt"
FACE_MODEL = MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
EMBED_MODEL = MODELS_DIR / "openface_nn4.small2.v1.t7"

face_net = None
embed_net = None


def init_models():
    global face_net, embed_net

    missing = []
    if not FACE_PROTO.exists():
        missing.append(FACE_PROTO.name)
    if not FACE_MODEL.exists():
        missing.append(FACE_MODEL.name)
    if not EMBED_MODEL.exists():
        missing.append(EMBED_MODEL.name)

    if missing:
        print("❌ Missing model files:", missing)
        return

    try:
        face_net = cv2.dnn.readNetFromCaffe(str(FACE_PROTO), str(FACE_MODEL))
        embed_net = cv2.dnn.readNetFromTorch(str(EMBED_MODEL))
        print("✅ Models loaded successfully")
    except Exception as e:
        print("❌ Model load failed:", e)


init_models()


# =========================================================
# ENCODINGS
# =========================================================
def load_encodings():
    if not ENCODINGS_PATH.exists():
        return {}
    try:
        return joblib.load(ENCODINGS_PATH)
    except Exception:
        return {}


def save_encodings(data):
    joblib.dump(data, ENCODINGS_PATH)


known_encodings = load_encodings()


# =========================================================
# UTIL
# =========================================================
def today_strings():
    t = date.today()
    return t.strftime("%Y-%m-%d"), t.strftime("%d %B %Y")


def clamp_box(x1, y1, x2, y2, w, h):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    return x1, y1, x2, y2


# =========================================================
# FACE RECOGNITION
# =========================================================
def detect_faces(frame, conf_thresh=0.7):
    if face_net is None:
        return []

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
    face_net.setInput(blob)
    detections = face_net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf >= conf_thresh:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype(int)
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes


def get_embedding(face_bgr):
    if embed_net is None:
        return None
    try:
        face = cv2.resize(face_bgr, (96, 96))
        blob = cv2.dnn.blobFromImage(face, 1 / 255.0, (96, 96), swapRB=True)
        embed_net.setInput(blob)
        return embed_net.forward().flatten()
    except Exception:
        return None


def recognize(emb, threshold=0.7):
    if emb is None or not known_encodings:
        return None

    best_roll, best_dist = None, float("inf")
    for roll, known in known_encodings.items():
        known = np.array(known, dtype=np.float32)
        d = np.linalg.norm(emb - known)
        if d < best_dist:
            best_roll, best_dist = roll, d

    return best_roll if best_dist < threshold else None


# =========================================================
# DB CRUD
# =========================================================
def db_list_students():
    db = get_db()
    rows = db.execute("SELECT roll, name, course, section FROM students ORDER BY roll").fetchall()
    db.close()
    return rows


def db_get_student(roll):
    db = get_db()
    row = db.execute("SELECT roll, name, course, section FROM students WHERE roll = ?", (roll,)).fetchone()
    db.close()
    return row


def db_upsert_student(roll, name, course="", section=""):
    db = get_db()
    db.execute("""
        INSERT INTO students(roll, name, course, section)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(roll) DO UPDATE SET
          name=excluded.name,
          course=excluded.course,
          section=excluded.section
    """, (roll, name, course, section))
    db.commit()
    db.close()


def db_delete_student(roll):
    db = get_db()
    db.execute("DELETE FROM attendance WHERE roll = ?", (roll,))
    db.execute("DELETE FROM students WHERE roll = ?", (roll,))
    db.commit()
    db.close()


def db_mark_attendance(roll, att_date, mode="auto"):
    student = db_get_student(roll)
    if not student:
        return False, "Student not found"

    db = get_db()
    now_time = datetime.now().strftime("%H:%M:%S")
    try:
        db.execute(
            "INSERT INTO attendance(roll, att_date, att_time, mode) VALUES(?, ?, ?, ?)",
            (roll, att_date, now_time, mode),
        )
        db.commit()
        ok = True
    except sqlite3.IntegrityError:
        ok = False
    finally:
        db.close()

    return ok, ("Attendance marked" if ok else "Already marked for this date")


def db_read_attendance(att_date):
    db = get_db()
    rows = db.execute("""
        SELECT
            a.id,
            s.name AS name,
            a.roll AS roll,
            s.course AS course,
            s.section AS section,
            a.att_time AS time,
            a.mode AS mode
        FROM attendance a
        JOIN students s ON s.roll = a.roll
        WHERE a.att_date = ?
        ORDER BY a.att_time
    """, (att_date,)).fetchall()
    db.close()
    return rows


def db_delete_attendance_row(att_id):
    db = get_db()
    db.execute("DELETE FROM attendance WHERE id = ?", (att_id,))
    db.commit()
    db.close()


def db_total_students():
    db = get_db()
    n = db.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    db.close()
    return int(n)


def db_present_count(att_date):
    db = get_db()
    n = db.execute("SELECT COUNT(DISTINCT roll) FROM attendance WHERE att_date = ?", (att_date,)).fetchone()[0]
    db.close()
    return int(n)


def db_class_days():
    db = get_db()
    rows = db.execute("SELECT DISTINCT att_date FROM attendance ORDER BY att_date").fetchall()
    db.close()
    return [r["att_date"] for r in rows]


def db_student_presence_days(roll):
    db = get_db()
    n = db.execute("SELECT COUNT(DISTINCT att_date) FROM attendance WHERE roll = ?", (roll,)).fetchone()[0]
    db.close()
    return int(n)


def db_trend_counts():
    db = get_db()
    rows = db.execute("""
        SELECT att_date, COUNT(DISTINCT roll) AS present
        FROM attendance
        GROUP BY att_date
        ORDER BY att_date
    """).fetchall()
    db.close()
    return rows


# =========================================================
# FACE CAPTURE
# =========================================================
def capture_face_for_roll(roll):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False

    success = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)
        if faces:
            x1, y1, x2, y2 = faces[0]
            face = frame[y1:y2, x1:x2]
            emb = get_embedding(face)

            if emb is not None:
                known_encodings[roll] = emb
                save_encodings(known_encodings)
                success = True
                break

        # press q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return success


# =========================================================
# CAMERA STREAM (AUTO MARK TODAY)
# =========================================================
def gen_frames():
    cap = cv2.VideoCapture(0)
    stable_roll = None
    stable_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = detect_faces(frame)
        if faces:
            x1, y1, x2, y2 = faces[0]
            face = frame[y1:y2, x1:x2]
            emb = get_embedding(face)
            roll = recognize(emb)

            if roll == stable_roll:
                stable_count += 1
            else:
                stable_roll, stable_count = roll, 1

            if stable_roll and stable_count >= 5:
                today_iso, _ = today_strings()
                db_mark_attendance(stable_roll, today_iso, mode="auto")
                stable_roll, stable_count = None, 0

            label = roll if roll else "Unknown"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buffer.tobytes() + b"\r\n")

    cap.release()


# =========================================================
# ROUTES
# =========================================================
@app.route("/")
def home():
    today_iso, today_pretty = today_strings()
    selected_date = request.args.get("date", today_iso)

    students = db_list_students()
    attendance = db_read_attendance(selected_date)

    total_students = len(students)
    present = db_present_count(selected_date)
    absent = max(total_students - present, 0)
    percent = round((present / total_students) * 100, 2) if total_students else 0.0

    model_ready = (face_net is not None and embed_net is not None and len(known_encodings) > 0)

    # Overall per-student % based on class days (days where at least one record exists)
    days = db_class_days()
    total_days = len(days) if days else 0

    per_student = []
    for s in students:
        present_days = db_student_presence_days(s["roll"])
        pct = round((present_days / total_days) * 100, 2) if total_days else 0.0
        per_student.append({
            "roll": s["roll"],
            "name": s["name"],
            "course": s["course"],
            "section": s["section"],
            "present_days": present_days,
            "total_days": total_days,
            "pct": pct
        })

    return render_template(
        "home.html",
        today_pretty=today_pretty,
        today_iso=today_iso,
        selected_date=selected_date,

        students=students,
        total_students=total_students,

        attendance=attendance,
        attendance_count=len(attendance),

        present=present,
        absent=absent,
        percent=percent,

        model_ready=model_ready,
        per_student=per_student
    )


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/register", methods=["POST"])
def register():
    roll = request.form["roll"].strip()
    name = request.form["name"].strip()
    course = request.form.get("course", "").strip()
    section = request.form.get("section", "").strip()

    if not roll or not name:
        flash("Roll and Name are required.", "danger")
        return redirect(url_for("home"))

    db_upsert_student(roll, name, course, section)
    flash("Student saved. Now click 'Capture Face' in Students list.", "success")
    return redirect(url_for("home"))


@app.route("/update_student/<roll>", methods=["POST"])
def update_student(roll):
    name = request.form.get("name", "").strip()
    course = request.form.get("course", "").strip()
    section = request.form.get("section", "").strip()

    if not name:
        flash("Name cannot be empty.", "danger")
        return redirect(url_for("home"))

    db_upsert_student(roll, name, course, section)
    flash("Student updated.", "success")
    return redirect(url_for("home"))


@app.route("/delete_student/<roll>", methods=["POST"])
def delete_student(roll):
    db_delete_student(roll)
    known_encodings.pop(roll, None)
    save_encodings(known_encodings)
    flash("Student deleted.", "success")
    return redirect(url_for("home"))


@app.route("/capture_face/<roll>", methods=["POST"])
def capture_face(roll):
    ok = capture_face_for_roll(roll)
    if ok:
        flash("Face captured successfully!", "success")
    else:
        flash("Face capture failed. Ensure camera works & face is visible.", "danger")
    return redirect(url_for("home"))


@app.route("/mark_manual", methods=["POST"])
def mark_manual():
    roll = request.form.get("roll_manual")
    selected_date = request.form.get("date_manual")
    if not selected_date:
        selected_date, _ = today_strings()

    ok, msg = db_mark_attendance(roll, selected_date, mode="manual")
    flash(msg, "success" if ok else "info")
    return redirect(url_for("home", date=selected_date))


@app.route("/delete_attendance/<int:att_id>", methods=["POST"])
def delete_attendance(att_id):
    selected_date = request.form.get("date")
    db_delete_attendance_row(att_id)
    flash("Attendance deleted.", "success")
    return redirect(url_for("home", date=selected_date))


@app.route("/download_excel")
def download_excel():
    day = request.args.get("date")
    if not day:
        day, _ = today_strings()

    rows = db_read_attendance(day)
    if not rows:
        flash("No attendance found for selected date.", "warning")
        return redirect(url_for("home", date=day))

    df = pd.DataFrame([dict(r) for r in rows]).rename(columns={
        "name": "Name", "roll": "Roll", "course": "Course", "section": "Section",
        "time": "Time", "mode": "Mode"
    })

    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name=f"attendance_{day}.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/download_pdf")
def download_pdf():
    day = request.args.get("date")
    if not day:
        day, _ = today_strings()

    students = db_list_students()
    attendance = db_read_attendance(day)

    total_students = len(students)
    present = db_present_count(day)
    percent = round((present / total_students) * 100, 2) if total_students else 0.0

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    y = h - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Face Attendance Report (UEL)")
    y -= 22

    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Date: {day}")
    y -= 16
    c.drawString(50, y, f"Students: {total_students} | Present: {present} | Attendance %: {percent}%")
    y -= 22

    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y, "Name")
    c.drawString(220, y, "Roll")
    c.drawString(320, y, "Time")
    c.drawString(400, y, "Mode")
    y -= 12
    c.line(50, y, w - 50, y)
    y -= 14

    c.setFont("Helvetica", 10)
    for r in attendance:
        if y < 80:
            c.showPage()
            y = h - 60
            c.setFont("Helvetica-Bold", 10)
            c.drawString(50, y, "Name")
            c.drawString(220, y, "Roll")
            c.drawString(320, y, "Time")
            c.drawString(400, y, "Mode")
            y -= 12
            c.line(50, y, w - 50, y)
            y -= 14
            c.setFont("Helvetica", 10)

        c.drawString(50, y, str(r["name"])[:28])
        c.drawString(220, y, str(r["roll"])[:12])
        c.drawString(320, y, str(r["time"])[:10])
        c.drawString(400, y, str(r["mode"])[:10])
        y -= 14

    c.showPage()
    c.save()
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=f"attendance_{day}.pdf", mimetype="application/pdf")


# ---------------- VISUALIZATIONS ----------------
@app.route("/attendance_pie")
def attendance_pie():
    day = request.args.get("date")
    if not day:
        day, _ = today_strings()

    total = db_total_students()
    present = db_present_count(day)
    absent = max(total - present, 0)

    plt.figure(figsize=(5, 4))
    if total == 0:
        plt.pie([1], labels=["No Students"])
    else:
        labels = [f"Present ({present})", f"Absent ({absent})"]
        plt.pie([present, absent], labels=labels, autopct="%1.1f%%")
    plt.title(f"Attendance % (Date: {day})")
    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format="png")
    plt.close()
    img.seek(0)
    return send_file(img, mimetype="image/png")


@app.route("/attendance_trend")
def attendance_trend():
    rows = db_trend_counts()

    plt.figure(figsize=(7, 4))
    if not rows:
        plt.title("Attendance Trend (Date-wise)")
        plt.xlabel("Date")
        plt.ylabel("Students Present")
        plt.tight_layout()
    else:
        dates = [r["att_date"] for r in rows]
        counts = [r["present"] for r in rows]
        plt.plot(dates, counts, marker="o")
        plt.xlabel("Date")
        plt.ylabel("Students Present")
        plt.title("Attendance Trend (Date-wise)")
        plt.xticks(rotation=45)
        plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format="png")
    plt.close()
    img.seek(0)
    return send_file(img, mimetype="image/png")


@app.route("/student_percent")
def student_percent():
    students = db_list_students()
    days = db_class_days()
    total_days = len(days) if days else 0

    names, percents = [], []
    for s in students:
        present_days = db_student_presence_days(s["roll"])
        pct = (present_days / total_days) * 100 if total_days else 0
        names.append(s["name"])
        percents.append(pct)

    plt.figure(figsize=(8, 4))
    plt.bar(names, percents)
    plt.ylabel("Attendance %")
    plt.title("Per-Student Attendance % (Overall)")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 100)
    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format="png")
    plt.close()
    img.seek(0)
    return send_file(img, mimetype="image/png")


# =========================================================
if __name__ == "__main__":
    print("🚀 Face Attendance System Running")
    app.run(host="0.0.0.0", port=5000, debug=True)