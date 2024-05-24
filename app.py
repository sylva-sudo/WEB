from flask import Flask, Response, jsonify, redirect, render_template, session, request, url_for
import mysql.connector
import cv2
from PIL import Image
import numpy as np
import os
from datetime import date
from flask_cors import CORS
import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
CORS(app)
 
cnt = 0
pause_cnt = 0
justscanned = False
 
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="flask_db"
)
mycursor = mydb.cursor()

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Generate dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def generate_dataset(nbr):
    face_classifier = cv2.CascadeClassifier("D:/TUGAS AKHIR/WEB/resources/haarcascade_frontalface_default.xml")
    eye_classifier = cv2.CascadeClassifier("D:/TUGAS AKHIR/WEB/resources/haarcascade_eye.xml")
    mouth_classifier = cv2.CascadeClassifier("D:/TUGAS AKHIR/WEB/resources/mouth.xml")

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        eyes = eye_classifier.detectMultiScale(gray, 1.3, 5)
        mouths = mouth_classifier.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return None, None, None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
            cropped_eyes = [(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes if x < ex < x + w and y < ey < y + h]
            cropped_mouths = [(mx, my, mw, mh) for (mx, my, mw, mh) in mouths if x < mx < x + w and y < my < y + h]
            return cropped_face, cropped_eyes, cropped_mouths
        return None, None, None

    cap = cv2.VideoCapture(0)

    mycursor.execute("SELECT IFNULL(MAX(img_id), 0) FROM img_dataset")
    row = mycursor.fetchone()
    lastid = row[0]

    img_id = lastid
    max_imgid = img_id + 100
    count_img = 0

    while True:
        ret, img = cap.read()
        face, eyes, mouths = face_cropped(img)
        if face is not None:
            count_img += 1
            img_id += 1
            face = cv2.resize(face, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = f"dataset/{nbr}.{img_id}.jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count_img), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            mycursor.execute("INSERT INTO img_dataset (img_id, img_person) VALUES (%s, %s)", (img_id, nbr))
            mydb.commit()

            for i, (ex, ey, ew, eh) in enumerate(eyes):
                eye = img[ey:ey + eh, ex:ex + ew]
                eye = cv2.resize(eye, (50, 50))
                eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(f"dataset/{nbr}_eye_{img_id}_{i}.jpg", eye)

            for j, (mx, my, mw, mh) in enumerate(mouths):
                mouth = img[my:my + mh, mx:mx + mw]
                mouth = cv2.resize(mouth, (100, 50))
                mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(f"dataset/{nbr}_mouth_{img_id}_{j}.jpg", mouth)

            frame = cv2.imencode('.jpg', face)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(1) == 13 or img_id >= max_imgid:
                break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/train_classifier/<nbr>')
def train_classifier(nbr):
    dataset_dir = "D:/TUGAS AKHIR/WEB/dataset"

    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".jpg") and "_" not in f]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)

    # Latih classifier dan simpan
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")

    return redirect('/')


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def face_recognition():
    def draw_boundary(img, faceCascade, eyeCascade, mouthCascade, scaleFactor, minNeighbors, color, text, clf):
        global justscanned, pause_cnt, cnt

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

        pause_cnt += 1
        coords = []

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            roi_gray = gray_image[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            eyes = eyeCascade.detectMultiScale(roi_gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
            mouth = mouthCascade.detectMultiScale(roi_gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=(int(w * 0.3), int(h * 0.1)), maxSize=(int(w * 0.7), int(h * 0.3)))

            # print(f"Detected eyes: {eyes}")
            # print(f"Detected mouth: {mouth}")

            if len(eyes) >= 2 and len(mouth) >= 1:  # Verifikasi bahwa mata dan mulut terdeteksi
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                for (mx, my, mw, mh) in mouth:
                    # Verifikasi bahwa mulut berada di bawah mata
                    if my > min([ey for (ex, ey, ew, eh) in eyes]):
                        cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (255, 0, 0), 2)
                        break  # biasanya hanya satu mulut

                id, pred = clf.predict(roi_gray)
                confidence = int(100 * (1 - pred / 300))

                if confidence > 70 and not justscanned:
                    mycursor.execute("SELECT a.img_person, b.prs_name, b.prs_skill FROM img_dataset a LEFT JOIN prs_mstr b ON a.img_person = b.prs_nbr WHERE img_id = %s", (id,))
                    row = mycursor.fetchone()

                    if row is not None:
                        pnbr = row[0]
                        pname = row[1]
                        pskill = row[2]

                        cnt += 1
                        n = (100 / 30) * cnt
                        w_filled = (cnt / 30) * w

                        cv2.putText(img, str(int(n))+' %', (x + 20, y + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 255, 255), 2, cv2.LINE_AA)
                        cv2.rectangle(img, (x, y + h + 40), (x + w, y + h + 50), color, 2)
                        cv2.rectangle(img, (x, y + h + 40), (x + int(w_filled), y + h + 50), (153, 255, 255), cv2.FILLED)

                        if cnt == 30:
                            cnt = 0
                            mycursor.execute("INSERT INTO accs_hist (accs_date, accs_prsn) VALUES (%s, %s)", (str(datetime.date.today()), pnbr))
                            mydb.commit()

                            cv2.putText(img, pname + ' | ' + pskill, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 255, 255), 2, cv2.LINE_AA)
                            justscanned = True
                            pause_cnt = 0
                else:
                    if not justscanned:
                        cv2.putText(img, 'UNKNOWN', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    if pause_cnt > 80:
                        justscanned = False

            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade, eyeCascade, mouthCascade):
        coords = draw_boundary(img, faceCascade, eyeCascade, mouthCascade, 1.2, 7, (255, 255, 0), "Face", clf)
        return img

    faceCascade = cv2.CascadeClassifier("D:/TUGAS AKHIR/WEB/resources/haarcascade_frontalface_default.xml")
    eyeCascade = cv2.CascadeClassifier("D:/TUGAS AKHIR/WEB/resources/haarcascade_eye.xml")
    mouthCascade = cv2.CascadeClassifier("D:/TUGAS AKHIR/WEB/resources/mouth.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    wCam, hCam = 400, 400
    
    cap = cv2.VideoCapture(0)
    # rtsp_link = "rtsp://your_rtsp_link"
    # cap = cv2.VideoCapture(rtsp_link)
    
    cap.set(3, wCam)
    cap.set(4, hCam)

    while True:
        ret, img = cap.read()
        img = recognize(img, clf, faceCascade, eyeCascade, mouthCascade)

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    
#>>>>>>>>>>>>>>>>>>>>>>>>>>>ROUTE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
@app.route('/')
def home():
    mycursor.execute("SELECT prs_nbr, prs_name, prs_skill, prs_active, prs_added FROM prs_mstr")
    data = mycursor.fetchall()
    if 'username' in session:
        return render_template('index.html', username=session['username'], data=data)
    else:
        return render_template('index.html', data=data)

 
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cur = mydb.cursor()
        cur.execute("SELECT username, password FROM user_adm WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()
        if user and password == user[1]:
            session['username'] = user[0]
            return redirect(url_for('home'))
        else:
            error = "Invalid username or password!"
            return render_template('login.html', error=error)
   
    return render_template('login.html')
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        cur = mydb.cursor()
        cur.execute("INSERT INTO user_adm (username, email, password) VALUES (%s, %s, %s)", (username, email, password))
        mydb.commit()
        cur.close()
        session['message'] = 'Registration successful. You can now login.'
        return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))
        
@app.route('/user')
def user():
    mycursor.execute("select prs_nbr, prs_name, prs_skill, prs_active, prs_added from prs_mstr")
    data = mycursor.fetchall()
 
    return render_template('user.html', data=data)

@app.route('/registration')
def addprsn():
    mycursor.execute("select ifnull(max(prs_nbr) + 1, 101) from prs_mstr")
    row = mycursor.fetchone()
    nbr = row[0]
    # print(int(nbr))
 
    return render_template('addprsn.html', newnbr=int(nbr))
 
@app.route('/addprsn_submit', methods=['POST'])
def addprsn_submit():
    prsnbr = request.form.get('txtnbr')
    prsname = request.form.get('txtname')
    prsskill = request.form.get('optskill')
 
    mycursor.execute("""INSERT INTO `prs_mstr` (`prs_nbr`, `prs_name`, `prs_skill`) VALUES
                    ('{}', '{}', '{}')""".format(prsnbr, prsname, prsskill))
    mydb.commit()
 
    # return redirect(url_for('home'))
    return redirect(url_for('vfdataset_page', prs=prsnbr))
 
@app.route('/vfdataset_page/<prs>')
def vfdataset_page(prs):
    return render_template('gendataset.html', prs=prs)
 
@app.route('/vidfeed_dataset/<nbr>')
def vidfeed_dataset(nbr):
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(generate_dataset(nbr), mimetype='multipart/x-mixed-replace; boundary=frame')
 
 
@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(face_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')
 
@app.route('/fr_page')
def fr_page():
    """Video streaming home page."""
    mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.prs_skill, a.accs_added "
                     "  from accs_hist a "
                     "  left join prs_mstr b on a.accs_prsn = b.prs_nbr "
                     " where a.accs_date = curdate() "
                     " order by 1 desc")
    data = mycursor.fetchall()
 
    return render_template('fr_page.html', data=data)
 
 
@app.route('/countTodayScan')
def countTodayScan():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="flask_db"
    )
    mycursor = mydb.cursor()
 
    mycursor.execute("select count(*) "
                     "  from accs_hist "
                     " where accs_date = curdate() ")
    row = mycursor.fetchone()
    rowcount = row[0]
 
    return jsonify({'rowcount': rowcount})
 
 
@app.route('/loadData', methods = ['GET', 'POST'])
def loadData(): 
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="flask_db"
    )
    mycursor = mydb.cursor()
 
    mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.prs_skill, date_format(a.accs_added, '%H:%i:%s') "
                     "  from accs_hist a "
                     "  left join prs_mstr b on a.accs_prsn = b.prs_nbr "
                     " where a.accs_date = curdate() "
                     " order BY a.accs_id DESC"
                     " limit 10")
    data = mycursor.fetchall()
 
    return jsonify(response = data)

@app.route('/loadDataLog', methods = ['GET', 'POST'])
def loadDataLog(): 
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="flask_db"
    )
    mycursor = mydb.cursor()
 
    mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.prs_skill, date_format(a.accs_added, '%d-%m-%Y %H:%i:%s') "
                     "  from accs_hist a "
                     "  left join prs_mstr b on a.accs_prsn = b.prs_nbr "
                     " order BY a.accs_id DESC")
    data = mycursor.fetchall()
 
    return jsonify(response = data)

@app.route('/log')
def log():
    mycursor.execute("select accs_id, accs_prsn, accs_date, accs_added from accs_hist")
    data = mycursor.fetchall()
 
    return render_template('log.html', data=data)




 
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
