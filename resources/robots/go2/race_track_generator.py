import xml.etree.ElementTree as xml_et
from pathlib import Path
import numpy as np
import cv2
import noise
import os

ROBOT = "go2"
INPUT_SCENE_PATH = os.path.join(os.path.dirname(__file__), "flat.xml")
OUTPUT_SCENE_PATH = os.path.join(os.path.dirname(__file__), "race_track_tmp.xml")
PATH_DIR = Path(__file__).parent.absolute()
# zyx euler angle to quaternion
def euler_to_quat(roll, pitch, yaw):
    cx = np.cos(roll / 2)
    sx = np.sin(roll / 2)
    cy = np.cos(pitch / 2)
    sy = np.sin(pitch / 2)
    cz = np.cos(yaw / 2)
    sz = np.sin(yaw / 2)

    return np.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ],
        dtype=np.float64,
    )


# zyx euler angle to rotation matrix
def euler_to_rot(roll, pitch, yaw):
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ],
        dtype=np.float64,
    )

    rot_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    rot_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return rot_z @ rot_y @ rot_x


# 2d rotate
def rot2d(x, y, yaw):
    nx = x * np.cos(yaw) - y * np.sin(yaw)
    ny = x * np.sin(yaw) + y * np.cos(yaw)
    return nx, ny


# 3d rotate
def rot3d(pos, euler):
    R = euler_to_rot(euler[0], euler[1], euler[2])
    return R @ pos


def list_to_str(vec):
    return " ".join(str(s) for s in vec)


class TerrainGenerator:

    def __init__(self) -> None:
        self.scene = xml_et.parse(INPUT_SCENE_PATH)
        self.root = self.scene.getroot()
        self.worldbody = self.root.find("worldbody")
        self.asset = self.root.find("asset")
        self._add_wood_material()
        self._add_sponge_material()

    def _add_wood_material(self):
        # 1. 添加纹理 (Texture)
        # 这里使用内置的 "flat" 类型配上棕色，模拟一种简单的木头颜色。
        # 如果你有真实的木纹图片(比如 wood.png)，请将 type="2d" builtin="flat" 
        # 改为 type="2d" file="../wood.png"
        tex = xml_et.SubElement(self.asset, "texture")
        tex.attrib["name"] = "wood_tex"
        tex.attrib["type"] = "2d"
        tex.attrib["file"] = "./assets/wood.png"
        tex.attrib["rgb1"] = "0.6 0.4 0.2" # 棕色 (RGB)
        tex.attrib["width"] = "512"
        tex.attrib["height"] = "512"
        
        # 2. 添加材质 (Material)
        mat = xml_et.SubElement(self.asset, "material")
        mat.attrib["name"] = "wood_mat"    # 材质名称，后面 AddBox 要用
        mat.attrib["texture"] = "wood_tex" # 关联上面的纹理
        mat.attrib["specular"] = "0.2"     # 木头反光度较低
        mat.attrib["shininess"] = "0.1"    # 亮度较低
        mat.attrib["rgba"] = "1 1 1 1"

    def _add_sponge_material(self):
        # 1. 添加纹理 (Texture)
        # 因为没有图片文件，我们使用 builtin="flat" 来生成纯色纹理
        tex = xml_et.SubElement(self.asset, "texture")
        tex.attrib["name"] = "sponge_tex"
        tex.attrib["type"] = "2d"
        tex.attrib["builtin"] = "flat"     # 使用内置平面纹理，不需要 file 路径
        # tex.attrib["rgb1"] = "1.0 0.7 0.7" # 设置颜色：粉色 (参考图片颜色)
        tex.attrib["rgb1"] = "0.90196 0.83922 0.56471" # 设置颜色：黄色
        tex.attrib["width"] = "512"
        tex.attrib["height"] = "512"
        
        # 2. 添加材质 (Material)
        mat = xml_et.SubElement(self.asset, "material")
        mat.attrib["name"] = "mat_sponge"  # 材质名称，AddBox 中调用这个名字
        mat.attrib["texture"] = "sponge_tex"
        
        # 海绵的关键视觉特性：不反光、不油亮
        mat.attrib["specular"] = "0.1"    # 几乎没有镜面反射 (相比木头的0.2要低很多)
        mat.attrib["shininess"] = "0.1"   # 几乎没有光泽
        mat.attrib["rgba"] = "1 0.7 0.7 1"     # 叠加颜色，保持原样

    # Add Box to scene
    def AddBox(self,
               position=[1.0, 0.0, 0.0],
               euler=[0.0, 0.0, 0.0], 
               size=[0.1, 0.1, 0.1], 
               sponge=False):
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = "box"
        geo.attrib["size"] = list_to_str(
            0.5 * np.array(size))  # half size of box for mujoco
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)
        # === 修改部分开始 ===
        if sponge:
            # 1. 视觉：使用海绵材质 (假设你在 asset 中定义的名字叫 mat_sponge)
            geo.attrib["material"] = "mat_sponge" 
            # 2. 物理：solref 时间常数越大越软 (0.02 比较软, 默认约 0.002)
            geo.attrib["solref"] = "0.03 1"
            geo.attrib["priority"] = "1"
            geo.attrib["solmix"] = "1" 
            # 3. 摩擦：海绵通常摩擦力较大 (可选)
            geo.attrib["friction"] = "1.2 0.005 0.0001" 
        else:
            geo.attrib["material"] = "wood_mat"
            geo.attrib["friction"] = "0.5 0.005 0.0001" 
    
    def AddGeometry(self,
               position=[1.0, 0.0, 0.0],
               euler=[0.0, 0.0, 0.0], 
               size=[0.1, 0.1],geo_type="box"):
        
        # geo_type supports "plane", "sphere", "capsule", "ellipsoid", "cylinder", "box"
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = geo_type
        geo.attrib["size"] = list_to_str(
            0.5 * np.array(size))  # half size of box for mujoco
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)
        geo.attrib["material"] = "wood_mat"

    def AddStairs(self,
                  init_pos=[1.0, 0.0, 0.0],
                  yaw=0.0,
                  width=0.2,
                  height=0.15,
                  length=1.5,
                  stair_nums=10):

        local_pos = [0.0, 0.0, -0.5 * height]
        for i in range(stair_nums):
            local_pos[0] += width
            local_pos[2] += height
            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            self.AddBox([x + init_pos[0], y + init_pos[1], local_pos[2]],
                        [0.0, 0.0, yaw], [width, length, height])
            
    
    def AddDownStairs(self,
                     init_pos=[1.0, 0.0, 0.0],
                     yaw=0.0,
                     width=0.3,
                     height=0.15,
                     length=1.5,
                     stair_nums=10):

        # 从上向下生成台阶：第一个台阶中心在 +0.5*height，随后每步降低 height
        local_pos = [0.0, 0.0, 0.5 * height]
        for i in range(stair_nums):
            local_pos[0] += width
            local_pos[2] -= height
            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            self.AddBox([x + init_pos[0], y + init_pos[1], local_pos[2]],
                        [0.0, 0.0, yaw], [width, length, height])


    def AddStairsSeries(self,
                       init_pos=[1.0, 0.0, 0.0],
                       yaw=0.0,
                       width=0.3,
                       length=1.5,
                       stair_nums_up=6,
                       stair_nums_down=6,
                       start_height=0.05,
                       step_inc=0.03,
                       max_height=0.2,
                       flat_length=0.5):
        """生成一系列台阶：先上台阶（stair_nums_up），顶部有一段平地（flat_length），再下台阶（stair_nums_down）。
        每完成一对上/下台阶后，单步高度增加 step_inc，直到达到 max_height。

        参数说明：
        - init_pos: 底层起点（列表），序列沿局部 x 方向展开
        - yaw: 台阶朝向
        - width: 每级台阶在 x 方向的深度（步幅）
        - length: 台阶在 y 方向的宽度（和 AddStairs 一致）
        - stair_nums_up/down: 上/下台阶的级数
        - start_height: 第一对台阶的每级高度
        - step_inc: 每对增加的高度
        - max_height: 最大每级高度（包含）
        - flat_length: 顶部平地长度（沿 x）
        """

        # 保持 init_pos 不变（上/下台阶成对结束后回到同一基准高度）
        base_pos = np.array(init_pos, dtype=float)
        height = start_height

        # 平台长度至少为 1.0 米
        platform_length = max(flat_length, 1.0)

        # 迭代每一对台阶直到高度超限
        while height <= max_height + 1e-8:
            # --- 上台阶 ---
            # 上台阶第 i 级中心 z = base_z + (-0.5 + i) * height, i = 1..stair_nums_up
            for i in range(1, stair_nums_up + 1):
                center_x_local = i * width
                center_z = base_pos[2] + (-0.5 + i) * height
                x, y = rot2d(center_x_local, 0.0, yaw)
                self.AddBox([x + base_pos[0], y + base_pos[1], center_z],
                            [0.0, 0.0, yaw], [width, length, height])

            # 顶部平地：放在最后一级之后，平台长度至少 platform_length
            last_up_center_x = stair_nums_up * width
            last_up_top_surface = base_pos[2] + stair_nums_up * height  # 顶部平面高度
            flat_thickness = height  # 平地厚度，使用与台阶同高度以保证接触
            flat_center_local_x = last_up_center_x + width / 2.0 + platform_length / 2.0
            flat_center_z = last_up_top_surface + flat_thickness / 2.0
            x, y = rot2d(flat_center_local_x, 0.0, yaw)
            # 尺寸：在 x 方向用 platform_length, y 用 length, z 用 flat_thickness
            self.AddBox([x + base_pos[0], y + base_pos[1], flat_center_z],
                        [0.0, 0.0, yaw], [platform_length, length, flat_thickness])

            # --- 下台阶 ---
            # 平地末端 x
            flat_end_x = last_up_center_x + width / 2.0 + platform_length
            for j in range(1, stair_nums_down + 1):
                center_x_local = flat_end_x + width / 2.0 + (j - 1) * width
                # 第 j 级下台阶的中心 z = last_up_top_surface - 0.5*height - (j-1)*height
                center_z = last_up_top_surface - 0.5 * height - (j - 1) * height
                x, y = rot2d(center_x_local, 0.0, yaw)
                self.AddBox([x + base_pos[0], y + base_pos[1], center_z],
                            [0.0, 0.0, yaw], [width, length, height])

            # 下台阶之后也添加一段平地（连接到下一组上台阶），长度至少 platform_length
            seq_end_local_x = flat_end_x + stair_nums_down * width  # 这是最后一个下台阶的前缘 x
            post_flat_center_local_x = seq_end_local_x + platform_length / 2.0
            # 该平地应与下一组上台阶的起始高度对齐：其顶面与 base_z + height 对齐
            post_flat_center_z = base_pos[2] + height / 2.0
            x, y = rot2d(post_flat_center_local_x, 0.0, yaw)
            self.AddBox([x + base_pos[0], y + base_pos[1], post_flat_center_z],
                        [0.0, 0.0, yaw], [platform_length, length, flat_thickness])

            # 为下一对台阶准备：把 base_pos 在 x 方向平移到当前序列末端（post flat 末端），保持 z 不变
            seq_total_end_local_x = seq_end_local_x + platform_length
            # 计算下一组基准位移，使下一组上台阶第一级的前缘与当前 post-flat 的末端无缝对接
            # base_shift_local_x 为相对于当前 base 的局部 x 偏移
            overlap = 1e-3  # 以米为单位，微小重叠以避免可视缝隙
            base_shift_local_x = seq_total_end_local_x - width / 2.0 - overlap
            dx, dy = rot2d(base_shift_local_x, 0.0, yaw)
            base_pos[0] = base_pos[0] + dx
            base_pos[1] = base_pos[1] + dy

            # 增加单级高度
            height = round(height + step_inc, 8)
            

    def AddSuspendStairs(self,
                         init_pos=[1.0, 0.0, 0.0],
                         yaw=1.0,
                         width=0.2,
                         height=0.15,
                         length=1.5,
                         gap=0.1,
                         stair_nums=10):

        local_pos = [0.0, 0.0, -0.5 * height]
        for i in range(stair_nums):
            local_pos[0] += width
            local_pos[2] += height
            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            self.AddBox([x + init_pos[0], y + init_pos[1], local_pos[2]],
                        [0.0, 0.0, yaw],
                        [width, length, abs(height - gap)])

    def AddRoughGround(self,
                       init_pos=[1.0, 0.0, 0.0],
                       euler=[0.0, -0.0, 0.0],
                       nums=[10, 10],
                       box_size=[0.5, 0.5, 0.5],
                       box_euler=[0.0, 0.0, 0.0],
                       separation=[0.2, 0.2],
                       box_size_rand=[0.05, 0.05, 0.05],
                       box_euler_rand=[0.2, 0.2, 0.2],
                       separation_rand=[0.05, 0.05]):

        local_pos = [0.0, 0.0, -0.5 * box_size[2]]
        new_separation = np.array(separation) + np.array(
            separation_rand) * np.random.uniform(-1.0, 1.0, 2)
        for i in range(nums[0]):
            local_pos[0] += new_separation[0]
            local_pos[1] = 0.0
            for j in range(nums[1]):
                new_box_size = np.array(box_size) + np.array(
                    box_size_rand) * np.random.uniform(-1.0, 1.0, 3)
                new_box_euler = np.array(box_euler) + np.array(
                    box_euler_rand) * np.random.uniform(-1.0, 1.0, 3)
                new_separation = np.array(separation) + np.array(
                    separation_rand) * np.random.uniform(-1.0, 1.0, 2)

                local_pos[1] += new_separation[1]
                pos = rot3d(local_pos, euler) + np.array(init_pos)
                self.AddBox(pos, new_box_euler, new_box_size)

    def AddPerlinHeighField(
            self,
            position=[1.0, 0.0, 0.0],  # position
            euler=[0.0, -0.0, 0.0],  # attitude
            size=[1.0, 1.0],  # width and length
            height_scale=0.2,  # max height
            negative_height=0.2,  # height in the negative direction of z axis
            image_width=128,  # height field image size
            img_height=128,
            smooth=100.0,  # smooth scale
            perlin_octaves=6,  # perlin noise parameter
            perlin_persistence=0.5,
            perlin_lacunarity=2.0,
            output_hfield_image="height_field.png"):

        # Generating height field based on perlin noise
        terrain_image = np.zeros((img_height, image_width), dtype=np.uint8)
        for y in range(image_width):
            for x in range(image_width):
                # Perlin noise
                noise_value = noise.pnoise2(x / smooth,
                                            y / smooth,
                                            octaves=perlin_octaves,
                                            persistence=perlin_persistence,
                                            lacunarity=perlin_lacunarity)
                terrain_image[y, x] = int((noise_value + 1) / 2 * 255)

        cv2.imwrite(str(PATH_DIR / "assets" / output_hfield_image), terrain_image)

        hfield = xml_et.SubElement(self.asset, "hfield")
        hfield.attrib["name"] = "perlin_hfield"
        hfield.attrib["size"] = list_to_str(
            [size[0] / 2.0, size[1] / 2.0, height_scale, negative_height])
        hfield.attrib["file"] = "../" + output_hfield_image

        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["type"] = "hfield"
        geo.attrib["hfield"] = "perlin_hfield"
        geo.attrib["pos"] = list_to_str(position)
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

    def AddHeighFieldFromImage(
            self,
            position=[1.0, 0.0, 0.0],  # position
            euler=[0.0, -0.0, 0.0],  # attitude
            size=[2.0, 1.6],  # width and length
            height_scale=0.02,  # max height
            negative_height=0.1,  # height in the negative direction of z axis
            input_img=None,
            output_hfield_image="height_field.png",
            image_scale=[1.0, 1.0],  # reduce image resolution
            invert_gray=False):

        input_image = cv2.imread(input_img)  # 替换为你的图像文件路径

        width = int(input_image.shape[1] * image_scale[0])
        height = int(input_image.shape[0] * image_scale[1])
        resized_image = cv2.resize(input_image, (width, height),
                                   interpolation=cv2.INTER_AREA)
        terrain_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        if invert_gray:
            terrain_image = 255 - position
        cv2.imwrite(str(PATH_DIR / "assets" / output_hfield_image), terrain_image)

        hfield = xml_et.SubElement(self.asset, "hfield")
        hfield.attrib["name"] = "image_hfield"
        hfield.attrib["size"] = list_to_str(
            [size[0] / 2.0, size[1] / 2.0, height_scale, negative_height])
        hfield.attrib["file"] = "../" + output_hfield_image

        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["type"] = "hfield"
        geo.attrib["hfield"] = "image_hfield"
        geo.attrib["pos"] = list_to_str(position)
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

    def Save(self):
        self.scene.write(OUTPUT_SCENE_PATH)

    def AddSlope(self,
                 position=[1.0, 0.0, 0.0],
                 yaw=0.0,
                 length=0.575,
                 width=0.55,
                 height=0.15,
                 thickness=0,
                 add_baffle=True,
                 add_baffle_height=False,
                 sponge=False):
        """
        生成带后背板和两侧挡板的斜坡。
        侧挡板上沿与斜坡面对齐。
        """
        if not thickness:
            thickness = 0.05       # 斜坡面板厚度
        side_thickness = 0.05  # 侧板厚度
        
        # ===========================
        # 1. 计算公共几何参数
        # ===========================
        
        # 计算坡度角和斜边长
        angle = np.arctan2(height, length)
        ramp_len = np.sqrt(length**2 + height**2)
        
        # 法线向量 (nx, nz)
        nx = np.sin(angle)
        nz = -np.cos(angle)
        
        # 斜坡面板的几何中心 (局部坐标)
        # 这里的逻辑是将面板中心沿着法线向下偏移厚度的一半，保证上表面对齐理想斜面
        mid_x = length / 2.0
        mid_z = height / 2.0
        lx = mid_x + nx * (thickness / 2.0)
        lz = mid_z + nz * (thickness / 2.0)
        
        # 统一的旋转角度 (Pitch: -angle, Yaw: yaw)
        final_euler = [0.0, -angle, yaw]

        # ===========================
        # 2. 生成主斜坡面 (Ramp)
        # ===========================
        gx, gy = rot2d(lx, 0, yaw)
        ramp_pos = [position[0] + gx, position[1] + gy, position[2] + lz]
        self.AddBox(ramp_pos, final_euler, [ramp_len, width, thickness],sponge=sponge)

        # ===========================
        # 3. 生成垂直背板 (Back Wall)
        # ===========================
        # 位于斜坡末端，高度为 height
        if add_baffle:
            back_lx = length + (thickness / 2.0)
            if add_baffle_height:
                back_lz = height
                height *= 2
            else:
                back_lz = height / 2.0
            bgx, bgy = rot2d(back_lx, 0, yaw)
            back_pos = [position[0] + bgx, position[1] + bgy, position[2] + back_lz]
            # 背板竖直放置，只受Yaw影响
            self.AddBox(back_pos, [0, 0, yaw], [thickness, width, height])

            # 生成两侧挡板 (Side Walls)
            side_h = height / 2
        
            # 计算侧板的中心 Z 坐标 (side_lz)
            # 目标：侧板的上表面 Z = 斜坡的上表面 Z
            # 斜坡上表面 Z (局部) = lz + thickness/2
            # 侧板上表面 Z (局部) = side_lz + side_h/2
            # 等式：lz + thickness/2 = side_lz + side_h/2
            # 解得：
            side_lz = lz + (thickness / 2.0) - (side_h / 2.0)
            if add_baffle_height:
                side_lz += side_h
                # side_h *= 2
                # lx += height * 2 / length / 2
            # 侧板的 X 坐标与斜坡中心一致 (lx)
            
            # 侧板的 Y 偏移量
            # 放在斜坡宽度的两侧：(斜坡宽/2) + (侧板厚/2)
            y_shift = (width / 2.0) + (side_thickness / 2.0)

            # 生成左右两个侧板
            for sign in [-1, 1]: # -1:左侧, 1:右侧
                if add_baffle_height and sign == 1: continue
                local_y = sign * y_shift
                
                # 将 (lx, local_y) 旋转 Yaw 角到全局
                sgx, sgy = rot2d(lx, local_y, yaw)
                
                side_pos = [
                    position[0] + sgx,
                    position[1] + sgy,
                    position[2] + side_lz
                ]
                
                # 侧板的旋转角度与斜坡完全一致，这样上边缘才会平行
                self.AddBox(side_pos, final_euler, [ramp_len, side_thickness, side_h])

    def AddSlopeGroup(self, position=[0.0, 0.0, 0.0], yaw=0.0, add_baffle=True, add_baffle_height=False,):
        
        L = 0.6  # 坡长 (爬升方向)
        W = 0.6   # 坡宽 (侧向)
        H = 0.164 
        
        p1_local = [-L / 2, -W]
        yaw1 = np.pi / 2
        
        p2_local = [-L, W/2]
        yaw2 = 0
        
        p3_local = [L, -W/2]
        yaw3 = np.pi
        
        p4_local = [L / 2, W]
        yaw4 = np.pi * 3 / 2

        blocks = [
            (p1_local, yaw1),
            (p2_local, yaw2),
            (p3_local, yaw3),
            (p4_local, yaw4)
        ]

        for pos_local, local_yaw in blocks:
            off_x, off_y = rot2d(pos_local[0], pos_local[1], yaw)
            
            abs_pos = [
                position[0] + off_x,
                position[1] + off_y,
                position[2]
            ]
            
            abs_yaw = yaw + local_yaw
            
            self.AddSlope(position=abs_pos,
                          yaw=abs_yaw,
                          length=L,
                          width=W,
                          height=H,
                          add_baffle=add_baffle,
                          add_baffle_height=add_baffle_height)
            
    def AddBlockyHeightField(
            self,
            position=[1.0, 0.0, 0.0],
            euler=[0.0, -0.0, 0.0],
            size=[1.0, 1.0],
            height_scale=0.2,       # 高度差幅度
            negative_height=0.1,
            image_width=128,
            img_height=128,
            smooth=50.0,            # 注意：如果要完全随机，把这个数改得很小（如 2.0）
            pixels_per_block=16,    # <--- 新参数：决定方块的大小
            output_hfield_image="height_field.png"):

        # 1. 准备图像数据
        terrain_image = np.zeros((img_height, image_width), dtype=np.uint8)
        
        # 2. 预先生成一个随机种子偏移，保证每次地形不一样
        seed_offset_x = np.random.randint(0, 10000)
        seed_offset_y = np.random.randint(0, 10000)

        for y in range(img_height):
            for x in range(image_width):
                
                # === 核心修改开始 ===
                # 这里的整除逻辑 (//) 是制造“方块感”的关键
                # 它将坐标强制归整，例如 x=0到15 都会变成 0，x=16到31 都会变成 16
                # 这样这 16 个像素取到的噪声值就是一模一样的，形成平坦的台阶
                block_x = (x // pixels_per_block) * pixels_per_block
                block_y = (y // pixels_per_block) * pixels_per_block
                
                # 使用归整后的 block_x, block_y 来生成噪声
                noise_value = noise.pnoise2((block_x + seed_offset_x) / smooth,
                                            (block_y + seed_offset_y) / smooth,
                                            octaves=1,          # 减少细节，让方块表面平整
                                            persistence=0.5,
                                            lacunarity=2.0)
                # === 核心修改结束 ===

                # 映射到 0-255
                terrain_image[y, x] = int((noise_value + 1) / 2 * 255)

        hfield = xml_et.SubElement(self.asset, "hfield")
        hfield.attrib["name"] = "perlin_hfield"
        hfield.attrib["size"] = list_to_str(
            [size[0] / 2.0, size[1] / 2.0, height_scale, negative_height])
        hfield.attrib["file"] = output_hfield_image

        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["type"] = "hfield"
        geo.attrib["hfield"] = "perlin_hfield"
        geo.attrib["pos"] = list_to_str(position)
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)
            
    def unit1_sponge(self):
        self.AddSlope(position=[0.0, 1.2, 0.0], yaw=np.pi, length=2.32, width=1.2, height=0.6, add_baffle_height=True)
        self.AddSlope(position=[-2.32, 0.0, 0.0], yaw=0, length=2.32, width=1.2, height=0.6, add_baffle_height=True)

        self.AddBox(position=[0.6, -0.6, 0.3], size=[1.2, 0.6, 0.03], euler=[np.pi / 2, 0, 0])
        self.AddBox(position=[-2.92, -0.6, 0.3], size=[1.2, 0.6, 0.03], euler=[np.pi / 2, 0, 0])
        self.AddBox(position=[0.6, 1.8, 0.3], size=[1.2, 0.6, 0.03], euler=[np.pi / 2, 0, 0])
        self.AddBox(position=[-2.92, 1.8, 0.3], size=[1.2, 0.6, 0.03], euler=[np.pi / 2, 0, 0])
        self.AddBox(position=[-3.52, 0.0, 0.3], size=[0.6, 1.2, 0.03], euler=[0, np.pi / 2, 0])

        self.AddBox(position=[-2.92, 0.6, 0.025], size=[1.2, 2.4, 0.05], sponge=True)
        self.AddBox(position=[0.6, 0.6, 0.025], size=[1.2, 2.4, 0.05], sponge=True)

        self.AddSlope(position=[0.0, 1.2, 0.05], yaw=np.pi, length=2.32, width=1.2, thickness=0.05, height=0.6, sponge=True, add_baffle=False)
        self.AddSlope(position=[-2.32, 0.0, 0.05], yaw=0, length=2.32, width=1.2, thickness=0.05, height=0.6, sponge=True, add_baffle=False)
        


    def unit2_slopes(self):
        self.AddBox(position=[1.225, 0.6, 0.0], size=[2.4, 0.05, 0.1], euler=[0, 0, np.pi / 2])
        self.AddSlopeGroup(position=[1.85, 0.0, 0.0], yaw=0.0, add_baffle=True)
        self.AddSlopeGroup(position=[1.85, 1.2, 0.0], yaw=0.0, add_baffle=True)
        self.AddBox(position=[2.475, 0.6, 0.0], size=[2.4, 0.05, 0.1], euler=[0, 0, np.pi / 2])
        self.AddSlopeGroup(position=[3.1, 0.0, 0.0], yaw=0.0, add_baffle=True)
        self.AddSlopeGroup(position=[3.1, 1.2, 0.0], yaw=0.0, add_baffle=True)
        self.AddSlopeGroup(position=[4.3, 0.0, 0.0], yaw=0.0, add_baffle=True)
        self.AddSlopeGroup(position=[4.3, 1.2, 0.0], yaw=0.0, add_baffle=True)
        self.AddBox(position=[3.7, 0.575, 0.0], size=[2.4, 0.05, 0.1])
        self.AddBox(position=[3.7, 0.625, 0.0], size=[2.4, 0.05, 0.1])
        self.AddBox(position=[4.925, 0.6, 0.0], size=[2.4, 0.05, 0.1], euler=[0, 0, np.pi / 2])
        self.AddSlopeGroup(position=[5.55, 0.0, 0.0], yaw=0.0, add_baffle=True)
        self.AddSlopeGroup(position=[5.55, 1.2, 0.0], yaw=0.0, add_baffle=True)
        self.AddBox(position=[6.175, 0.6, 0.0], size=[2.4, 0.05, 0.1], euler=[0, 0, np.pi / 2])
        self.AddBox(position=[1.9, 1.8, 0.3], size=[1.4, 0.6, 0.03], euler=[np.pi / 2, 0, 0])
        self.AddBox(position=[3.2, 1.8, 0.3], size=[1.2, 0.6, 0.03], euler=[np.pi / 2, 0, 0])
        self.AddBox(position=[4.4, 1.8, 0.3], size=[1.2, 0.6, 0.03], euler=[np.pi / 2, 0, 0])
        self.AddBox(position=[5.6, 1.8, 0.3], size=[1.2, 0.6, 0.03], euler=[np.pi / 2, 0, 0])
        
        self.AddBox(position=[1.9, -0.6, 0.3], size=[1.4, 0.6, 0.03], euler=[np.pi / 2, 0, 0])
        self.AddBox(position=[3.2, -0.6, 0.3], size=[1.2, 0.6, 0.03], euler=[np.pi / 2, 0, 0])
        self.AddBox(position=[4.4, -0.6, 0.3], size=[1.2, 0.6, 0.03], euler=[np.pi / 2, 0, 0])
        self.AddBox(position=[5.6, -0.6, 0.3], size=[1.2, 0.6, 0.03], euler=[np.pi / 2, 0, 0])

        self.AddBox(position=[1.2, 0.0, 0.3], size=[0.6, 1.2, 0.03], euler=[0, np.pi / 2, 0])
        self.AddBox(position=[2.5, 1.2, 0.3], size=[0.6, 1.2, 0.03], euler=[0, np.pi / 2, 0])
        self.AddBox(position=[4.95, 0.0, 0.3], size=[0.6, 1.2, 0.03], euler=[0, np.pi / 2, 0])
        self.AddBox(position=[6.2, 1.2, 0.3], size=[0.6, 1.2, 0.03], euler=[0, np.pi / 2, 0])


    def unit3_stairs(self):
        self.AddBox(position = [6.8, 1.2, 0.1], size = [1.2, 1.2, 0.2])
        self.AddBox(position = [8.0, 0.0, 0.1], size = [1.2, 1.2, 0.2])
        self.AddBox(position = [9.2, 1.2, 0.1], size = [1.2, 1.2, 0.2])
        self.AddBox(position = [10.4, 0.0, 0.1], size = [1.2, 1.2, 0.2])

        self.AddBox(position=[6.8, 1.8, 0.3], size=[1.2, 0.6, 0.03], euler=[np.pi / 2, 0, 0])
        self.AddBox(position=[8.0, 1.8, 0.3], size=[1.2, 0.6, 0.03], euler=[np.pi / 2, 0, 0])
        self.AddBox(position=[9.2, 1.8, 0.3], size=[1.2, 0.6, 0.03], euler=[np.pi / 2, 0, 0])
        self.AddBox(position=[10.4, 1.8, 0.3], size=[1.2, 0.6, 0.03], euler=[np.pi / 2, 0, 0])
        
        self.AddBox(position=[6.8, -0.6, 0.3], size=[1.2, 0.6, 0.03], euler=[np.pi / 2, 0, 0])
        self.AddBox(position=[8.0, -0.6, 0.3], size=[1.2, 0.6, 0.03], euler=[np.pi / 2, 0, 0])
        self.AddBox(position=[9.2, -0.6, 0.3], size=[1.2, 0.6, 0.03], euler=[np.pi / 2, 0, 0])
        self.AddBox(position=[10.4, -0.6, 0.3], size=[1.2, 0.6, 0.03], euler=[np.pi / 2, 0, 0])

        self.AddBox(position=[11.0, 1.2, 0.3], size=[0.6, 1.2, 0.03], euler=[0, np.pi / 2, 0])

    def unit4_diagonal(self):
        self.AddSlope(position=[12.8, 1.8, 0.0], yaw=np.pi / 2, length=2.32, width=1.2, height=0.6, add_baffle_height=True)
        self.AddSlope(position=[12.8, 1.8, 0.05], yaw=np.pi / 2, length=2.32/48, width=1.2, height=0.6/48, thickness=0.05, add_baffle=False)
        self.AddSlope(position=[12.8, 4.07, 0.6375], yaw=np.pi / 2, length=2.32/48, width=1.2, height=0.6/48, thickness=0.05, add_baffle=False)
        self.AddSlope(position=[12.225, 1.8, 0.05], yaw=np.pi / 2, length=2.32, width=0.05, height=0.6, thickness=0.05, add_baffle=False)
        self.AddSlope(position=[13.375, 1.8, 0.05], yaw=np.pi / 2, length=2.32, width=0.05, height=0.6, thickness=0.05, add_baffle=False)
        self.AddBox(position=[12.8, 3.54, 0.475], size=[1.64, 0.05, 0.05], euler=[0.0, -0.1779, 0.7685])
        self.AddBox(position=[12.8, 2.38, 0.175], size=[1.64, 0.05, 0.05], euler=[0.0, 0.1779, -0.7685])

        self.AddSlope(position=[11.6, 4.12, 0.0], yaw=-np.pi / 2, length=2.32, width=1.2, height=0.6, add_baffle_height=True)
        self.AddSlope(position=[11.6, 4.12, 0.05], yaw=-np.pi / 2, length=2.32/48, width=1.2, height=0.6/48, thickness=0.05, add_baffle=False)
        self.AddSlope(position=[11.6, 1.85, 0.6375], yaw=-np.pi / 2, length=2.32/48, width=1.2, height=0.6/48, thickness=0.05, add_baffle=False)
        self.AddSlope(position=[12.175, 4.12, 0.05], yaw=-np.pi / 2, length=2.32, width=0.05, height=0.6, thickness=0.05, add_baffle=False)
        self.AddSlope(position=[11.025, 4.12, 0.05], yaw=-np.pi / 2, length=2.32, width=0.05, height=0.6, thickness=0.05, add_baffle=False)
        self.AddBox(position=[11.6, 3.54, 0.175], size=[1.64, 0.05, 0.05], euler=[0.0, -0.1779, -0.7685])
        self.AddBox(position=[11.6, 2.38, 0.475], size=[1.64, 0.05, 0.05], euler=[0.0, 0.1779, 0.7685])

        self.AddBox(position=[11.6, -0.6, 0.3], size=[1.2, 0.6, 0.03], euler=[np.pi / 2, 0, 0])
        self.AddBox(position=[12.2, 0.0, 0.3], size=[0.6, 1.2, 0.03], euler=[0, np.pi / 2, 0])
        self.AddBox(position=[12.8, 0.6, 0.3], size=[1.2, 0.6, 0.03], euler=[np.pi / 2, 0, 0])
        self.AddBox(position=[13.4, 1.2, 0.3], size=[0.6, 1.2, 0.03], euler=[0, np.pi / 2, 0])

        self.AddBox(position=[12.2, 0.625, 0.025], size=[2.4, 0.05, 0.05])
        self.AddBox(position=[12.2, 1.775, 0.025], size=[2.4, 0.05, 0.05])
        self.AddBox(position=[11.025, 1.2, 0.025], size=[1.2, 0.05, 0.05], euler=[0, 0, np.pi / 2])
        self.AddBox(position=[13.375, 1.2, 0.025], size=[1.2, 0.05, 0.05], euler=[0, 0, np.pi / 2])
        self.AddBox(position=[11.6, 1.2, 0.025], size=[1.64, 0.05, 0.05], euler=[0, 0, -np.pi / 4])
        self.AddBox(position=[12.8, 1.2, 0.025], size=[1.64, 0.05, 0.05], euler=[0, 0, np.pi / 4])

        self.AddBox(position=[12.2, 4.145, 0.025], size=[2.4, 0.05, 0.05])
        self.AddBox(position=[12.2, 5.295, 0.025], size=[2.4, 0.05, 0.05])
        self.AddBox(position=[11.025, 4.72, 0.025], size=[1.2, 0.05, 0.05], euler=[0, 0, np.pi / 2])
        self.AddBox(position=[13.375, 4.72, 0.025], size=[1.2, 0.05, 0.05], euler=[0, 0, np.pi / 2])
        self.AddBox(position=[11.6, 4.72, 0.025], size=[1.64, 0.05, 0.05], euler=[0, 0, np.pi / 4])
        self.AddBox(position=[12.8, 4.72, 0.025], size=[1.64, 0.05, 0.05], euler=[0, 0, -np.pi / 4])

        self.AddBox(position=[11.0, 4.72, 0.3], size=[0.6, 1.2, 0.03], euler=[0, np.pi / 2, 0])
        self.AddBox(position=[13.4, 4.72, 0.3], size=[0.6, 1.2, 0.03], euler=[0, np.pi / 2, 0])
        self.AddBox(position=[11.6, 5.32, 0.3], size=[1.2, 0.6, 0.03], euler=[np.pi / 2, 0, 0])

        self.AddBox(position=[11.6, 0.0, 0.0], size=[1.2, 1.2, 0.03])
        self.AddBox(position=[12.2, 1.2, 0.0], size=[2.4, 1.2, 0.03])
        self.AddBox(position=[12.2, 4.72, 0.0], size=[2.4, 1.2, 0.03])


    def unit5_sandstone(self):

        self.AddBox(position=[11.0, 5.92, 0.3], size=[0.6, 1.2, 0.03], euler=[0, np.pi / 2, 0])
        self.AddBox(position=[11.0, 7.12, 0.3], size=[0.6, 1.2, 0.03], euler=[0, np.pi / 2, 0])
        self.AddBox(position=[11.0, 8.32, 0.3], size=[0.6, 1.2, 0.03], euler=[0, np.pi / 2, 0])
        self.AddBox(position=[11.0, 9.52, 0.3], size=[0.6, 1.2, 0.03], euler=[0, np.pi / 2, 0])
        self.AddBox(position=[13.4, 5.92, 0.3], size=[0.6, 1.2, 0.03], euler=[0, np.pi / 2, 0])
        self.AddBox(position=[13.4, 7.12, 0.3], size=[0.6, 1.2, 0.03], euler=[0, np.pi / 2, 0])
        self.AddBox(position=[13.4, 8.32, 0.3], size=[0.6, 1.2, 0.03], euler=[0, np.pi / 2, 0])
        self.AddBox(position=[13.4, 9.52, 0.3], size=[0.6, 1.2, 0.03], euler=[0, np.pi / 2, 0])

        self.AddBox(position=[12.8, 6.52, 0.3], size=[1.2, 0.6, 0.03], euler=[np.pi / 2, 0, 0])
        self.AddBox(position=[12.8, 10.12, 0.3], size=[1.2, 0.6, 0.03], euler=[np.pi / 2, 0, 0])
        self.AddBox(position=[11.6, 8.92, 0.3], size=[1.2, 0.6, 0.03], euler=[np.pi / 2, 0, 0])

        self.AddBox(position=[12.2, 6.545, 0.05], size=[2.4, 0.05, 0.05])
        self.AddBox(position=[12.2, 8.895, 0.05], size=[2.4, 0.05, 0.05])
        self.AddBox(position=[11.025, 7.72, 0.05], size=[2.4, 0.05, 0.05], euler=[0, 0, np.pi / 2])
        self.AddBox(position=[13.375, 7.72, 0.05], size=[2.4, 0.05, 0.05], euler=[0, 0, np.pi / 2])
        self.AddBox(position=[12.2, 7.72, 0.05], size=[3.28, 0.05, 0.05], euler=[0, 0, -np.pi / 4])
        self.AddBox(position=[12.2, 7.72, 0.05], size=[3.28, 0.05, 0.05], euler=[0, 0, np.pi / 4])
        self.AddBox(position=[11.6, 7.12, 0.05], size=[1.64, 0.05, 0.05], euler=[0, 0, -np.pi / 4])
        self.AddBox(position=[12.8, 7.12, 0.05], size=[1.64, 0.05, 0.05], euler=[0, 0, np.pi / 4])
        self.AddBox(position=[11.6, 8.32, 0.05], size=[1.64, 0.05, 0.05], euler=[0, 0, np.pi / 4])
        self.AddBox(position=[12.8, 8.32, 0.05], size=[1.64, 0.05, 0.05], euler=[0, 0, -np.pi / 4])

        self.AddBlockyHeightField(position=[12.2, 7.72, -0.00], size=[2.4, 4.8], height_scale=0.08)


if __name__ == "__main__":
    tg = TerrainGenerator()

    # # Box obstacle
    # tg.AddBox(position=[0.55 / 2 + 2, 0, 0.075], size=[np.sqrt(0.55 * 0.55 + 0.15*0.15), 0.1, 0.01], euler=[np.pi / 2, 0, 0])
    # tg.AddBox(position=[3.5, 0, 0.075], size=[1.5, 0.5, 0.01], euler=[0, 0, 0])
    
    # # Geometry obstacle
    # # geo_type supports "plane", "sphere", "capsule", "ellipsoid", "cylinder", "box"
    # tg.AddGeometry(position=[1.5, 0.0, 0.25], euler=[0, 0, 0.0], size=[1.0,0.5,0.5],geo_type="cylinder")

    # # Slope
    # tg.AddBox(position=[2.0, 2.0, 0.5],
    #           euler=[0.0, -0.5, 0.0],
    #           size=[3, 1.5, 0.1])

    # # Stairs
    # tg.AddStairs(init_pos=[1.0, 4.0, 0.0], yaw=0.0)

    # # Suspend stairs
    # tg.AddSuspendStairs(init_pos=[1.0, 6.0, 0.0], yaw=0.0)

    # # Rough ground
    # tg.AddRoughGround(init_pos=[-2.5, 5.0, 0.0],
    #                   euler=[0, 0, 0.0],
    #                   nums=[10, 8])

    # # Perlin heigh field
    # tg.AddPerlinHeighField(position=[-1.5, 4.0, 0.0], size=[2.0, 1.5])

    # # Heigh field from image
    # tg.AddHeighFieldFromImage(position=[-1.5, 2.0, 0.0],
    #                           euler=[0, 0, -1.57],
    #                           size=[2.0,2.0],
    #                           input_img="./unitree_robot.jpeg",
    #                           image_scale=[1.0, 1.0],
    #                           output_hfield_image="unitree_hfield.png")
    tg.unit1_sponge()
    tg.unit2_slopes()
    tg.unit3_stairs()
    tg.unit4_diagonal()
    tg.unit5_sandstone()

    tg.Save()
