import numpy as np
import OpenGL.GL  as gl
import OpenGL.GLU as glu


class Renderer:

    def __init__(self, width=640, height=480):

        self.width   = width
        self.height  = height
        self.objects = []

        self.camera_distance = 500.0
        self.camera_angle_x  = 20.0
        self.camera_angle_y  = 0.0

        self.wireframe  = False
        self._gl_ready  = False
        self._fbo       = None
        self._rbo_color = None
        self._rbo_depth = None


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


    def setup(self):

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glShadeModel(gl.GL_SMOOTH)

        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [1.0,  1.0, 1.0, 0.0])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE,  [1.0,  1.0, 1.0, 1.0])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT,  [0.3,  0.3, 0.3, 1.0])


    def set_projection(self):

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(45.0, self.width / self.height, 1.0, 3000.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)


    def render_to_image(self):

        if not self._gl_ready:
            self._init_gl()
            self.set_projection()

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)
        gl.glViewport(0, 0, self.width, self.height)

        self.render_frame(background_color=(0.0, 0.0, 0.0))

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


    def composite_onto(self, base_canvas, threshold=15):

        gl_img = self.render_to_image()

        if gl_img is None:
            return base_canvas

        mask   = np.any(gl_img > threshold, axis=2)
        result = base_canvas.copy()
        result[mask] = gl_img[mask]

        return result


    def render_frame(self, background_color=(0.15, 0.06, 0.06)):

        gl.glClearColor(*background_color, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()

        glu.gluLookAt(0, 0, self.camera_distance, 0, 0, 0, 0, 1, 0)
        gl.glRotatef(self.camera_angle_x, 1, 0, 0)
        gl.glRotatef(self.camera_angle_y, 0, 1, 0)

        for obj in self.objects:
            self._draw_mesh(obj)


    def _draw_mesh(self, obj):

        if len(obj.vertices) == 0:
            return

        gl.glPushMatrix()

        gl.glTranslatef(*obj.position)
        gl.glRotatef(obj.rotation[0], 1, 0, 0)
        gl.glRotatef(obj.rotation[1], 0, 1, 0)
        gl.glRotatef(obj.rotation[2], 0, 0, 1)
        gl.glScalef(*obj.scale)

        if obj.selected:
            gl.glColor3f(1.0, 0.85, 0.2)
        else:
            gl.glColor3f(*obj.color)

        if self.wireframe:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        else:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_NORMAL_ARRAY)

        gl.glVertexPointer(3, gl.GL_FLOAT, 0, obj.vertices)
        gl.glNormalPointer(gl.GL_FLOAT,    0, obj.normals)

        indices = np.array(obj.indices, dtype=np.uint32)
        gl.glDrawElements(
            gl.GL_TRIANGLES, len(indices), gl.GL_UNSIGNED_INT, indices
        )

        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisableClientState(gl.GL_NORMAL_ARRAY)

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glPopMatrix()


    def add_object(self, obj):
        self.objects.append(obj)

    def remove_object(self, obj):
        if obj in self.objects:
            self.objects.remove(obj)

    def clear_objects(self):
        self.objects.clear()

    def toggle_wireframe(self):
        self.wireframe = not self.wireframe