import numpy as np
import OpenGL.GL  as gl
import OpenGL.GLU as glu


class Renderer:

    def __init__(self, width=640, height=480):

        self.width   = width
        self.height  = height
        self.objects = []

        self.camera_distance = 500.0

        self._gl_ready  = False
        self._fbo       = None
        self._rbo_color = None
        self._rbo_depth = None


    # ─────────────────────────────
    def _init_gl(self):

        import OpenGL.GLUT as glut

        glut.glutInit()
        glut.glutInitDisplayMode(
            glut.GLUT_DOUBLE | glut.GLUT_RGB | glut.GLUT_DEPTH
        )
        glut.glutInitWindowSize(1, 1)
        glut.glutCreateWindow(b'ctx')
        glut.glutHideWindow()

        self._setup_fbo()
        self.setup()
        self._gl_ready = True


    # ─────────────────────────────
    def _setup_fbo(self):

        self._fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)

        self._rbo_color = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._rbo_color)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER, gl.GL_RGB8, self.width, self.height
        )
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
            gl.GL_RENDERBUFFER, self._rbo_color
        )

        self._rbo_depth = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._rbo_depth)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24,
            self.width, self.height
        )
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
            gl.GL_RENDERBUFFER, self._rbo_depth
        )

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)


    # ─────────────────────────────
    def setup(self):

        gl.glEnable(gl.GL_DEPTH_TEST)

        # restore lighting for shapes
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)

        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)

        gl.glShadeModel(gl.GL_SMOOTH)

        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [1,1,1,0])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE,  [1,1,1,1])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT,  [0.3,0.3,0.3,1])

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)


    def set_projection(self):

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(45.0, self.width/self.height, 1.0, 3000.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)


    # ─────────────────────────────
    def render_frame(self):

        gl.glClearColor(0,0,0,1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()

        glu.gluLookAt(0,0,self.camera_distance, 0,0,0, 0,1,0)

        for obj in self.objects:
            self._draw_mesh(obj)


    # ─────────────────────────────
    def render_to_image(self):

        if not self._gl_ready:
            self._init_gl()
            self.set_projection()

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)
        gl.glViewport(0, 0, self.width, self.height)

        self.render_frame()

        data = gl.glReadPixels(
            0, 0, self.width, self.height,
            gl.GL_BGR, gl.GL_UNSIGNED_BYTE
        )

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        img = np.frombuffer(data, dtype=np.uint8).reshape(
            (self.height, self.width, 3)
        )
        img = np.flipud(img)

        return img


    # ─────────────────────────────
    def composite_onto(self, base_canvas, threshold=5):

        gl_img = self.render_to_image()

        mask = np.any(gl_img > threshold, axis=2)

        result = base_canvas.copy()
        result[mask] = gl_img[mask]

        return result


    # ─────────────────────────────
    def _draw_mesh(self, obj):

        if getattr(obj, 'kind', None) == 'hand':
            self._draw_hand(obj)
            return

        if len(obj.vertices) == 0:
            return

        gl.glPushMatrix()

        gl.glTranslatef(*obj.position)
        gl.glRotatef(obj.rotation[0],1,0,0)
        gl.glRotatef(obj.rotation[1],0,1,0)
        gl.glRotatef(obj.rotation[2],0,0,1)
        gl.glScalef(*obj.scale)

        gl.glColor3f(*obj.color)

        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_NORMAL_ARRAY)

        gl.glVertexPointer(3, gl.GL_FLOAT, 0, obj.vertices)
        gl.glNormalPointer(gl.GL_FLOAT, 0, obj.normals)

        gl.glDrawElements(
            gl.GL_TRIANGLES,
            len(obj.indices),
            gl.GL_UNSIGNED_INT,
            obj.indices
        )

        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisableClientState(gl.GL_NORMAL_ARRAY)

        gl.glPopMatrix()


    # ─────────────────────────────
    # 🔥 HAND
    # ─────────────────────────────
    def _draw_hand(self, hand):

        if hand.landmarks is None:
            return

        pts = hand.landmarks.copy() * 1.25

        gl.glDisable(gl.GL_LIGHTING)

        # subtle sci-fi blue
        gl.glColor4f(0.45, 0.75, 0.9, 0.55)

        fingers = [
            [0,1,2,3,4],
            [0,5,6,7,8],
            [0,9,10,11,12],
            [0,13,14,15,16],
            [0,17,18,19,20]
        ]

        # 🔥 draw fingers
        for finger in fingers:
            self._draw_finger_mesh_clean(pts, finger, radius=5.5)

        # 🔥 clean palm (minimal, no webbing)
        palm_rows = [
            [0,5,9,13,17],
            [1,6,10,14,18],
            [2,7,11,15,19]
        ]

        for row in palm_rows:

            segments = [
                # ❌ REMOVE THIS → (row[0], row[1])  ← causes thumb-index web

                (row[1], row[2]),  # index → middle
                (row[2], row[3]),  # middle → ring
                (row[3], row[4])   # ring → pinky
            ]

            for a, b in segments:
                gl.glBegin(gl.GL_LINES)
                gl.glVertex3f(*pts[a])
                gl.glVertex3f(*pts[b])
                gl.glEnd()
        palm_columns = [
            [0,1,2],     # thumb side
            [5,6,7],
            [9,10,11],
            [13,14,15],
            [17,18,19]
        ]

        for col in palm_columns:
            for i in range(len(col)-1):
                a = col[i]
                b = col[i+1]

                gl.glBegin(gl.GL_LINES)
                gl.glVertex3f(*pts[a])
                gl.glVertex3f(*pts[b])
                gl.glEnd()
                gl.glEnable(gl.GL_LIGHTING)

    def _draw_finger_mesh_clean(self, pts, indices, radius=5.5, segments=3):

        for i in range(len(indices)-1):

            p1 = pts[indices[i]]
            p2 = pts[indices[i+1]]

            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length < 1e-5:
                continue

            direction /= length

            up = np.array([0,0,1])
            if abs(np.dot(direction, up)) > 0.9:
                up = np.array([1,0,0])

            side = np.cross(direction, up)
            side /= np.linalg.norm(side)
            up = np.cross(side, direction)

            t = i / (len(indices)-1)

            # 🔥 thicker, smooth finger shape
            r = radius * (0.75 + 0.35*(1 - t**1.2))

            # ───────────── ring ─────────────
            gl.glBegin(gl.GL_LINE_LOOP)
            for j in range(segments):
                angle = 2*np.pi*j/segments
                offset = np.cos(angle)*side*r + np.sin(angle)*up*r
                gl.glVertex3f(*(p1 + offset))
            gl.glEnd()

            # ───────────── minimal verticals ─────────────
            gl.glBegin(gl.GL_LINES)
            for j in range(0, segments, 2):
                angle = 2*np.pi*j/segments
                offset = np.cos(angle)*side*r + np.sin(angle)*up*r

                gl.glVertex3f(*(p1 + offset))
                gl.glVertex3f(*(p2 + offset))
            gl.glEnd()

        # 🔥 rounded fingertip (ONLY ONCE)
        tip = pts[indices[-1]]
        prev = pts[indices[-2]]

        direction = tip - prev
        length = np.linalg.norm(direction)

        if length > 1e-5:
            direction /= length

            up = np.array([0,0,1])
            if abs(np.dot(direction, up)) > 0.9:
                up = np.array([1,0,0])

            side = np.cross(direction, up)
            side /= np.linalg.norm(side)
            up = np.cross(side, direction)

            gl.glBegin(gl.GL_LINE_LOOP)
            for j in range(segments):
                angle = 2*np.pi*j/segments
                offset = np.cos(angle)*side*(radius*0.3) + np.sin(angle)*up*(radius*0.3)
                gl.glVertex3f(*(tip + offset))
            gl.glEnd()
        # ─────────────────────────────
        # 🔥 ROUNDED TIP
        # ─────────────────────────────
        tip = pts[indices[-1]]
        prev = pts[indices[-2]]

        direction = tip - prev
        length = np.linalg.norm(direction)

        if length > 1e-5:
            direction /= length

            up = np.array([0,0,1])
            if abs(np.dot(direction, up)) > 0.9:
                up = np.array([1,0,0])

            side = np.cross(direction, up)
            side /= np.linalg.norm(side)
            up = np.cross(side, direction)

            gl.glBegin(gl.GL_LINE_LOOP)
            for j in range(segments):
                angle = 2*np.pi*j/segments

                # shrinking radius → rounded cap
                offset = np.cos(angle)*side*(radius*0.3) + np.sin(angle)*up*(radius*0.3)

                gl.glVertex3f(*(tip + offset))

            gl.glEnd()

    # ─────────────────────────────
    def add_object(self, obj):
        self.objects.append(obj)

    def remove_object(self, obj):
        if obj in self.objects:
            self.objects.remove(obj)

    def clear_objects(self):
        self.objects.clear()