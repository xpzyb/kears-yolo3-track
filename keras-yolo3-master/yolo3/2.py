
def detect_video(yolo, video_path, output_path=""):

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    loop = 15
    frame = 0
    while True:
        return_value, img = vid.read()
        if not return_value:
            print ( "The video is over!" )

            return
        if frame % loop == 0:
            image = Image.fromarray(img)
            loc, scores, lab = yolo.detect_image(image)
            result = np.asarray(image)
            initimg = result
            yolo.close_session()
        else:
            loc, scores, lab = tracking ( image , img , loc, scores, lab, frame )
        cv2.imshow ( "result" , result )

        if isOutput:
            out.write ( img )

        if cv2.waitKey ( 1 ) & 0xFF == ord ( 'q' ):
            break

        # quit on ESC button

        if cv2.waitKey ( 1 ) & 0xFF == 27:  # Esc pressed

            break

        frame = frame + 1
        frame = frame % loop

tracker = cv2.MultiTracker_create ()

COLORS = np.random.randint ( 0 , 255 , size=(80 , 3) , dtype='uint8' )
def tracking(initImg, img, loc, scores, lab, frame):
    if frame == 1:
        global tracker  # 这步很重要，每次初始化跟踪时需要清除原先所跟踪的目标；否则，跟踪的目标会累加

        tracker = cv2.MultiTracker_create ()
        for i , newbox in enumerate (loc ):
          ok = tracker.add(cv2.TrackerKCF_create(), initImg, (newbox[0], newbox[1], newbox[2], newbox[3]))
          if not ok:
            print("The tracker initialization failed!")
            return
    ok, boxes = tracker.update(img)
    if ok:
        loc = boxes
        for i, box in enumerate(boxes):
            color = [int(c) for c in COLORS[i]]
            c1,c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(img, c1,c2, color, 2)
            text = '{}: {:.3f}'.format(lab[i], scores[i])
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (int(box[0]), int(box[1]) - text_h - baseline), (int(box[0]) + text_w, int(box[1])), color, -1)
            cv2.putText(img, text, (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return loc, scores, lab

