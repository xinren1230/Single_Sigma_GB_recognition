import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

# For misorientation calculations
from orix.quaternion import Orientation, Quaternion, symmetry
from orix.vector import Vector3d
from scipy.spatial.transform import Rotation as R
# Simple tooltip class for tkinter widgets

# Rotation state trackers (in degrees)
rotation_state_g1 = {'euler': None, 'rotations': [0, 0, 0], 'custom': 0.0}
rotation_state_g2 = {'euler': None, 'rotations': [0, 0, 0], 'custom': 0.0}

# Store Sigma axis after calculation
sigma_axis_global = None



def CreateToolTip(widget, text):
    """
    Create a tooltip for a given widget
    """
    tipwindow = None

    def showtip(event=None):
        nonlocal tipwindow
        if tipwindow or not text:
            return
        x = widget.winfo_rootx() + 20
        y = widget.winfo_rooty() + widget.winfo_height() + 1
        tipwindow = tw = tk.Toplevel(widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("tahoma", "8", "normal")
        )
        label.pack(ipadx=1)

    def hidetip(event=None):
        nonlocal tipwindow
        if tipwindow:
            tipwindow.destroy()
            tipwindow = None

    widget.bind("<Enter>", showtip)
    widget.bind("<Leave>", hidetip)
    
# 1. Core functions
def rotation_matrix_to_euler(rotation_matrix):
    return rotation_matrix.as_euler('ZXZ', degrees=True)

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def misorientation_angle(rot1, rot2):
    delta_rot = rot1.inv() * rot2
    angle = delta_rot.magnitude()
    return np.degrees(angle)

def crystal_rotation(euler_angles, axis, angle):
    initial_rotation = R.from_euler('ZXZ', euler_angles, degrees=True)
    crystal_axis = normalize_vector(axis)
    transformed_axis = initial_rotation.apply(crystal_axis)
    angle_radians = np.radians(angle)
    axis_rotation = R.from_rotvec(angle_radians * transformed_axis)
    new_rotation = axis_rotation * initial_rotation
    new_euler_angles = rotation_matrix_to_euler(new_rotation)
    return new_euler_angles

def compute_symmetry_reduced_orientation(ori1, ori2, symmetry_str='Oh'):
    ori1_quat = Quaternion.from_euler(ori1, degrees=True)
    ori1_quat.symmetry = getattr(symmetry, symmetry_str)
    ori2_quat = Quaternion.from_euler(ori2, degrees=True)
    ori2_quat.symmetry = getattr(symmetry, symmetry_str)
    symmetry_ori = ori1_quat.symmetry

    angles = []
    ori2_sym = symmetry_ori.outer(ori2_quat)
    misorientation = ori1_quat * ~ori2_sym

    for orien in misorientation:
        orim = Orientation(orien)
        angles.append(orim.angle)

    angles_array = np.array(angles)
    indices = angles_array.argmin(axis=0)
    out = np.take_along_axis(ori2_sym, indices[np.newaxis], axis=0).squeeze()
    ori2_sym_reduced = out.to_euler(degrees=True)
    return ori2_sym_reduced

def misorientation(rot1, rot2):
    delta_rot = rot1.inv() * rot2
    misorientation_axis = delta_rot.as_rotvec()
    angle = np.degrees(delta_rot.magnitude())
    if 180 - angle < angle:
        misorientation_axis = -misorientation_axis
        angle = 180 - angle
    return misorientation_axis, angle

def misorientation_calc(euler_angles1, euler_angles2):
    rot1 = R.from_euler('ZXZ', euler_angles1, degrees=True)
    rot2 = R.from_euler('ZXZ', euler_angles2, degrees=True)
    misorientation_axis, misorientation_ang = misorientation(rot1, rot2)
    return misorientation_axis, misorientation_ang

def are_axes_equivalent(axis2, axis1, angle_threshold=5.0):
    axis1 = axis1 / np.linalg.norm(axis1)
    axis2 = axis2 / np.linalg.norm(axis2)
    axis_permutations = []
    signs = [+1, -1]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if len({i, j, k}) == 3:
                    for s1 in signs:
                        for s2 in signs:
                            for s3 in signs:
                                perm = [s1 * axis1[i], s2 * axis1[j], s3 * axis1[k]]
                                axis_permutations.append(perm)
    for perm in axis_permutations:
        cos_angle = np.dot(perm, axis2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        if angle <= angle_threshold:
            return True, perm
    return False, None

def axis_angle_to_rotation_matrix(axis, angle_deg):
    angle = np.deg2rad(angle_deg)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    return np.array([
        [c + x*x*C,    x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,  c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,  z*y*C + x*s, c + z*z*C]
    ])

def is_sigma_boundary(axis, misorientation_angle, structure_type='FCC', angle_threshold=15.0, tolerance_widening=2):
    sigma_data = {
        '3': {'FCC': [{'axis': [1, 1, 1], 'angle': 60}],
              'BCC': [{'axis': [1, 1, 1], 'angle': 60}]},
        '5': {'FCC': [{'axis': [1, 0, 0], 'angle': 36.87}],
              'BCC': [{'axis': [1, 0, 0], 'angle': 36.87}]},
        '7': {'FCC': [{'axis': [1, 1, 1], 'angle': 38.21}],
              'BCC': [{'axis': [1, 1, 1], 'angle': 38.21}]},
        '9': {'FCC': [{'axis': [1, 1, 0], 'angle': 38.94}],
              'BCC': [{'axis': [1, 1, 0], 'angle': 38.94}]},
        '11': {'FCC': [{'axis': [1, 1, 0], 'angle': 50.48}],
               'BCC': [{'axis': [1, 1, 0], 'angle': 50.48}]},
        '13a': {'FCC': [{'axis': [1, 0, 0], 'angle': 22.62}],
                'BCC': [{'axis': [1, 0, 0], 'angle': 22.62}]},
        '13b': {'FCC': [{'axis': [1, 1, 1], 'angle': 27.79}],
                'BCC': [{'axis': [1, 1, 1], 'angle': 27.79}]},
        '15': {'FCC': [{'axis': [2, 1, 0], 'angle': 48.2}],
               'BCC': [{'axis': [2, 1, 0], 'angle': 48.2}]},
        '17a': {'FCC': [{'axis': [1, 0, 0], 'angle': 28.07}],
                'BCC': [{'axis': [1, 0, 0], 'angle': 28.07}]},
        '17b': {'FCC': [{'axis': [2, 2, 1], 'angle': 61.9}],
                'BCC': [{'axis': [2, 2, 1], 'angle': 61.9}]},
        '19a': {'FCC': [{'axis': [1, 1, 0], 'angle': 26.53}],
                'BCC': [{'axis': [1, 1, 0], 'angle': 26.53}]},
        '19b': {'FCC': [{'axis': [1, 1, 1], 'angle': 46.8}],
                'BCC': [{'axis': [1, 1, 1], 'angle': 46.8}]},
        '21a': {'FCC': [{'axis': [1, 1, 1], 'angle': 21.79}],
                'BCC': [{'axis': [1, 1, 1], 'angle': 21.79}]},
        '21b': {'FCC': [{'axis': [2, 1, 1], 'angle': 44.4}],
                'BCC': [{'axis': [2, 1, 1], 'angle': 44.4}]},
        '23': {'FCC': [{'axis': [3, 1, 1], 'angle': 40.5}],
               'BCC': [{'axis': [3, 1, 1], 'angle': 40.5}]},
        '25a': {'FCC': [{'axis': [1, 0, 0], 'angle': 16.3}],
                'BCC': [{'axis': [1, 0, 0], 'angle': 16.3}]},
        '25b': {'FCC': [{'axis': [3, 3, 1], 'angle': 51.7}],
                'BCC': [{'axis': [3, 3, 1], 'angle': 51.7}]},
        '27a': {'FCC': [{'axis': [1, 1, 0], 'angle': 31.59}],
                'BCC': [{'axis': [1, 1, 0], 'angle': 31.59}]},
        '27b': {'FCC': [{'axis': [2, 1, 0], 'angle': 35.43}],
                'BCC': [{'axis': [2, 1, 0], 'angle': 35.43}]},
        '29a': {'FCC': [{'axis': [1, 0, 0], 'angle': 43.6}],
                'BCC': [{'axis': [1, 0, 0], 'angle': 43.6}]},
        '29b': {'FCC': [{'axis': [2, 2, 1], 'angle': 46.4}],
                'BCC': [{'axis': [2, 2, 1], 'angle': 46.4}]},
        '31a': {'FCC': [{'axis': [1, 1, 1], 'angle': 17.9}],
                'BCC': [{'axis': [1, 1, 1], 'angle': 17.9}]},
        '31b': {'FCC': [{'axis': [2, 1, 1], 'angle': 52.2}],
                'BCC': [{'axis': [2, 1, 1], 'angle': 52.2}]},
        '33a': {'FCC': [{'axis': [1, 1, 0], 'angle': 20.0}],
                'BCC': [{'axis': [1, 1, 0], 'angle': 20.0}]},
        '33b': {'FCC': [{'axis': [3, 1, 1], 'angle': 33.6}],
                'BCC': [{'axis': [3, 1, 1], 'angle': 33.6}]},
        '33c': {'FCC': [{'axis': [1, 1, 0], 'angle': 59.0}],
                'BCC': [{'axis': [1, 1, 0], 'angle': 59.0}]},
        '35a': {'FCC': [{'axis': [2, 1, 1], 'angle': 34.0}],
                'BCC': [{'axis': [2, 1, 1], 'angle': 34.0}]},
        '35b': {'FCC': [{'axis': [3, 3, 1], 'angle': 43.2}],
                'BCC': [{'axis': [3, 3, 1], 'angle': 43.2}]},
        '37a': {'FCC': [{'axis': [1, 0, 0], 'angle': 18.9}],
                'BCC': [{'axis': [1, 0, 0], 'angle': 18.9}]},
        '37b': {'FCC': [{'axis': [3, 1, 0], 'angle': 43.1}],
                'BCC': [{'axis': [3, 1, 0], 'angle': 43.1}]},
        '37c': {'FCC': [{'axis': [1, 1, 1], 'angle': 50.6}],
                'BCC': [{'axis': [1, 1, 1], 'angle': 50.6}]},
        '39a': {'FCC': [{'axis': [1, 1, 1], 'angle': 32.2}],
                'BCC': [{'axis': [1, 1, 1], 'angle': 32.2}]},
        '39b': {'FCC': [{'axis': [3, 2, 1], 'angle': 50.1}],
                'BCC': [{'axis': [3, 2, 1], 'angle': 50.1}]},
        '41a': {'FCC': [{'axis': [1, 0, 0], 'angle': 12.7}],
                'BCC': [{'axis': [1, 0, 0], 'angle': 12.7}]},
        '41b': {'FCC': [{'axis': [2, 1, 0], 'angle': 40.9}],
                'BCC': [{'axis': [2, 1, 0], 'angle': 40.9}]},
        '41c': {'FCC': [{'axis': [1, 1, 0], 'angle': 55.9}],
                'BCC': [{'axis': [1, 1, 0], 'angle': 55.9}]},
        '43a': {'FCC': [{'axis': [1, 1, 1], 'angle': 15.2}],
                'BCC': [{'axis': [1, 1, 1], 'angle': 15.2}]},
        '43b': {'FCC': [{'axis': [2, 1, 0], 'angle': 27.9}],
                'BCC': [{'axis': [2, 1, 0], 'angle': 27.9}]},
        '43c': {'FCC': [{'axis': [3, 3, 2], 'angle': 60.8}],
                'BCC': [{'axis': [3, 3, 2], 'angle': 60.8}]},
        '45a': {'FCC': [{'axis': [3, 1, 1], 'angle': 28.6}],
                'BCC': [{'axis': [3, 1, 1], 'angle': 28.6}]},
        '45b': {'FCC': [{'axis': [2, 2, 1], 'angle': 36.9}],
                'BCC': [{'axis': [2, 2, 1], 'angle': 36.9}]},
        '45c': {'FCC': [{'axis': [2, 2, 1], 'angle': 53.1}],
                'BCC': [{'axis': [2, 2, 1], 'angle': 53.1}]},
        '47a': {'FCC': [{'axis': [3, 3, 1], 'angle': 37.1}],
                'BCC': [{'axis': [3, 3, 1], 'angle': 37.1}]},
        '47b': {'FCC': [{'axis': [3, 2, 0], 'angle': 43.7}],
                'BCC': [{'axis': [3, 2, 0], 'angle': 43.7}]},
        '49a': {'FCC': [{'axis': [1, 1, 1], 'angle': 43.6}],
                'BCC': [{'axis': [1, 1, 1], 'angle': 43.6}]},
        '49b': {'FCC': [{'axis': [5, 1, 1], 'angle': 43.6}],
                'BCC': [{'axis': [5, 1, 1], 'angle': 43.6}]},
        '49c': {'FCC': [{'axis': [3, 2, 2], 'angle': 49.2}],
                'BCC': [{'axis': [3, 2, 2], 'angle': 49.2}]}
    }
    

    axis = np.array(axis)

    for sigma, configs in sigma_data.items():
        if structure_type not in configs:
            continue
        for config in configs[structure_type]:
            predefined_axis = np.array(config['axis'])
            eq_result, eq_axis = are_axes_equivalent(axis, predefined_axis, angle_threshold)
            if eq_result:
                predefined_angle = config['angle']
                deviation = abs(misorientation_angle - predefined_angle)
                brandon_threshold = tolerance_widening * 15 / np.sqrt(float(sigma.rstrip('abc')))
                if deviation <= brandon_threshold:
                    return True, sigma, deviation, eq_axis, predefined_angle

    return False, None, None, None, None

# 2. Add Cube Drawing
def draw_cube(canvas, euler_angles, size=80, sigma_axis=None):
    canvas.delete("all")
    w, h = canvas.winfo_width(), canvas.winfo_height()
    cx, cy = w/2, h/2
    d = size/2
    axis_len = size*0.8

    # 1) Build the Bunge ZXZ rotation matrix
    phi1, Phi, phi2 = np.deg2rad(euler_angles)
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c,  s  = np.cos(Phi),  np.sin(Phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)
    Rz1 = np.array([[ c1,-s1,0],[ s1, c1,0],[  0,   0,1]])
    Rx  = np.array([[  1,   0,   0],[  0,   c,  -s],[  0,   s,   c]])
    Rz2 = np.array([[ c2,-s2,0],[ s2, c2,0],[  0,   0,   1]])
    Rm  = Rz1 @ Rx @ Rz2

    # helper to project a 3D point to 2D canvas coords
    def proj(pt3):
        return (pt3[0] + cx, -pt3[1] + cy)

    # 2) Draw global frame
    canvas.create_line(cx,cy, cx+axis_len,cy,
                       fill='red', width=2, arrow=tk.LAST)
    canvas.create_text(cx+axis_len+5, cy, text='X', fill='red', anchor='w')
    canvas.create_line(cx,cy, cx,cy-axis_len,
                       fill='blue', width=2, arrow=tk.LAST)
    canvas.create_text(cx, cy-axis_len-5, text='Y', fill='blue', anchor='s')
    rdot = 4
    canvas.create_oval(cx-rdot,cy-rdot, cx+rdot,cy+rdot,
                       fill='green', outline='')
    canvas.create_text(cx+rdot+3, cy+rdot+3, text='Z', fill='green', anchor='nw')

    # 3) Legend in top-left
    legend = [
        ('[100]', 'magenta'),
        ('[010]', 'orange'),
        ('[001]', 'teal'),
    ]
    lx, ly = 10, 10
    line_len = 20
    for i, (label, color) in enumerate(legend):
        y = ly + i*20
        canvas.create_line(lx, y, lx+line_len, y, fill=color, width=3)
        canvas.create_text(lx+line_len+5, y, text=label, anchor='w')

    # 4) Compute rotated cube verts & project
    verts = np.array([
        [-d,-d,-d],[-d,-d, d],[-d, d,-d],[-d, d, d],
        [ d,-d,-d],[ d,-d, d],[ d, d,-d],[ d, d, d]
    ])
    rot_verts = verts @ Rm.T
    proj2 = [proj(p) for p in rot_verts]

    # 5) Define and draw only the three edge‐groups
    faces = [[0,1,3,2],[4,6,7,5],[0,4,5,1],
             [2,3,7,6],[0,2,6,4],[1,5,7,3]]
    normals = [np.array([-1,0,0]),np.array([1,0,0]),
               np.array([0,-1,0]),np.array([0,1,0]),
               np.array([0,0,-1]),np.array([0,0,1])]
    face_vis = [(Rm @ n)[2] > 0 for n in normals]

    edges = {
        '[100]': [(0,4),(1,5),(2,6),(3,7)],
        '[010]': [(0,2),(1,3),(4,6),(5,7)],
        '[001]': [(0,1),(2,3),(4,5),(6,7)],
    }
    colors = {'[100]':'magenta','[010]':'orange','[001]':'teal'}

    for label, eds in edges.items():
        col = colors[label]
        for i,j in eds:
            adj = [fi for fi,face in enumerate(faces) if i in face and j in face]
            front = any(face_vis[fi] for fi in adj)
            x1,y1 = proj2[i]; x2,y2 = proj2[j]
            if front:
                canvas.create_line(x1,y1,x2,y2, fill=col, width=2)
            else:
                canvas.create_line(x1,y1,x2,y2, fill=col,
                                   width=1, dash=(4,2))
    # 6) Optional: draw Sigma rotation axis (if provided)
    if sigma_axis is not None:
        axis_norm = sigma_axis / np.linalg.norm(sigma_axis)
        axis_len = size * 1.2
        start = np.array([0, 0, 0])
        end = axis_len * axis_norm

        # Apply same rotation matrix to axis
        start_rot = start @ Rm.T
        end_rot = end @ Rm.T
        x1, y1 = proj(start_rot)
        x2, y2 = proj(end_rot)

        canvas.create_line(x1, y1, x2, y2, fill='black', width=3, arrow=tk.LAST)
        canvas.create_text(x2+5, y2+5, text='Σ axis', fill='black', anchor='nw')

# 3. GUI part
def calculate_misorientation():
    try:
        e1_str = entry_g1.get().strip()
        e2_str = entry_g2.get().strip()
        structure_type = combo_structure.get().strip()
        
                # Read new threshold inputs
        angle_threshold = float(entry_threshold.get().strip())
        tolerance_widening = float(entry_tolerance.get().strip())

        euler_grain1 = list(map(float, e1_str.split(',')))
        euler_grain2 = list(map(float, e2_str.split(',')))

        if len(euler_grain1) != 3 or len(euler_grain2) != 3:
            raise ValueError("Euler angles must be three comma-separated values each.")

        euler_grain2_sym_reduced = compute_symmetry_reduced_orientation(euler_grain1, euler_grain2)

        axis, misorientation_ang = misorientation_calc(euler_grain1, euler_grain2_sym_reduced)

        # Pass the user-specified thresholds
        is_sig, sigma_value, deviation, standard_axis, predefined_angle = is_sigma_boundary(
            axis[0], misorientation_ang,
            structure_type=structure_type,
            angle_threshold=angle_threshold,
            tolerance_widening=tolerance_widening
        )

        result = []
        result.append(f"Symmetry-reduced grain2: {np.round(euler_grain2_sym_reduced[0], 4)} deg")
        result.append(f"Misorientation axis (approx.): {np.round(axis[0], 4)}")
        result.append(f"Misorientation angle (deg): {np.round(misorientation_ang, 4)}")

        if is_sig:
            result.append(f"**This is a Σ{sigma_value} boundary**")
            result.append(f"Deviation from exact Σ boundary angle: {np.round(deviation, 4)} deg")
            result.append(f"Standard axis for Σ{sigma_value}: {standard_axis}")
            result.append(f"Standard angle for Σ{sigma_value}: {predefined_angle} deg")
                       # Optional: Check deviation to the perfect orientation
            # Recompute standard Euler angles for that boundary
            std_euler_angles = crystal_rotation(euler_grain1, standard_axis, predefined_angle)
            # Compare actual grain2 to standard orientation
            dev_axis, dev_angle = misorientation_calc(euler_grain2_sym_reduced, std_euler_angles)
            result.append(f"Additional deviation from exact orientation: {np.round(dev_angle,4)} deg, deviation axis: {np.round(dev_axis[0],4)}.")
        else:
            result.append("This is not a recognized low Σ boundary.")

            
        output_text.delete('1.0', tk.END)
        output_text.insert(tk.END, "\n".join(result))

        # Update cubes
        global sigma_axis_global
        if is_sig:
            sigma_axis_global = standard_axis
        else:
            sigma_axis_global = None

        draw_cube(canvas_g1, euler_grain1, sigma_axis=sigma_axis_global)
        draw_cube(canvas_g2, euler_grain2, sigma_axis=sigma_axis_global)



    except Exception as ex:
        messagebox.showerror("Error", str(ex))

        
# 3. GUI part
root = tk.Tk()
root.title("Grain Boundary Misorientation with Cubes")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

# Grain inputs
label_g1 = ttk.Label(frame, text="Grain1 Euler angles (deg, comma-separated):")
label_g1.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
entry_g1 = ttk.Entry(frame, width=30)
entry_g1.grid(row=0, column=1, padx=5, pady=5)
entry_g1.insert(0, "189.2,53.8,313.0")

label_g2 = ttk.Label(frame, text="Grain2 Euler angles (deg, comma-separated):")
label_g2.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
entry_g2 = ttk.Entry(frame, width=30)
entry_g2.grid(row=1, column=1, padx=5, pady=5)
entry_g2.insert(0, "246.0,67.9,104.4")

# New threshold inputs
label_threshold = ttk.Label(frame, text="Angle threshold (deg):")
label_threshold.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
entry_threshold = ttk.Entry(frame, width=10)
entry_threshold.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
entry_threshold.insert(0, "10")
CreateToolTip(label_threshold, "Maximum allowed deviation angle (°) between the rotation axis and the standard axis")

label_tolerance = ttk.Label(frame, text="Tolerance widening factor:")
label_tolerance.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
entry_tolerance = ttk.Entry(frame, width=10)
entry_tolerance.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
entry_tolerance.insert(0, "1")
CreateToolTip(label_tolerance, "Multiplier for allowable deviation range in the Brandon criterion")

# Crystal structure
label_struct = ttk.Label(frame, text="Crystal Structure (for Σ data):")
label_struct.grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
combo_structure = ttk.Combobox(frame, values=["FCC", "BCC"], width=5)
combo_structure.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)
combo_structure.current(0)

# Calculate button
btn_calc = ttk.Button(frame, text="Calculate", command=calculate_misorientation)
btn_calc.grid(row=5, column=0, columnspan=2, pady=10)

# Canvas for cubes
canvas_frame = ttk.Frame(root, padding="10")
canvas_frame.grid(row=1, column=0)
canvas_g1 = tk.Canvas(canvas_frame, width=200, height=200, bg='white')
canvas_g2 = tk.Canvas(canvas_frame, width=200, height=200, bg='white')
canvas_g1.grid(row=0, column=0, padx=10)
canvas_g2.grid(row=0, column=1, padx=10)

# Step input
label_step = ttk.Label(frame, text="Rotation step angle (deg):")
label_step.grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)
entry_step = ttk.Entry(frame, width=10)
entry_step.grid(row=6, column=1, padx=5, pady=5, sticky=tk.W)
entry_step.insert(0, "5")

# New Euler angle display
label_new_g1 = ttk.Label(frame, text="Updated Grain1 Euler angles:")
label_new_g1.grid(row=7, column=0, padx=5, pady=5, sticky=tk.W)
updated_g1 = ttk.Label(frame, text="---")
updated_g1.grid(row=7, column=1, padx=5, pady=5, sticky=tk.W)

label_new_g2 = ttk.Label(frame, text="Updated Grain2 Euler angles:")
label_new_g2.grid(row=8, column=0, padx=5, pady=5, sticky=tk.W)
updated_g2 = ttk.Label(frame, text="---")
updated_g2.grid(row=8, column=1, padx=5, pady=5, sticky=tk.W)


def reset_rotation(grain_id):
    global sigma_axis_global
    state = rotation_state_g1 if grain_id == 1 else rotation_state_g2
    state['rotations'] = [0, 0, 0]
    state['custom'] = 0.0

    # Reset display
    if grain_id == 1:
        updated_g1.config(text="---")
        custom_rot_g1.config(text="0.0°")
        rot_display_g1.config(text="0°, 0°, 0°")
        base_euler = list(map(float, entry_g1.get().strip().split(',')))
        draw_cube(canvas_g1, base_euler, sigma_axis=None)
    else:
        updated_g2.config(text="---")
        custom_rot_g2.config(text="0.0°")
        rot_display_g2.config(text="0°, 0°, 0°")
        base_euler = list(map(float, entry_g2.get().strip().split(',')))
        draw_cube(canvas_g2, base_euler, sigma_axis=None)

    # Also clear global sigma axis after reset
    sigma_axis_global = None

def update_sigma_axis_if_possible():
    global sigma_axis_global

    try:
        # Try to use updated Euler angles or fall back to entry values
        euler1 = list(map(float, updated_g1.cget("text").split(',')))
        euler2 = list(map(float, updated_g2.cget("text").split(',')))

        structure_type = combo_structure.get().strip()
        angle_threshold = float(entry_threshold.get().strip())
        tolerance_widening = float(entry_tolerance.get().strip())

        sym_euler2 = compute_symmetry_reduced_orientation(euler1, euler2)
        axis, mis_ang = misorientation_calc(euler1, sym_euler2)
        is_sig, sigma_value, deviation, std_axis, predefined_angle = is_sigma_boundary(
            axis[0], mis_ang,
            structure_type=structure_type,
            angle_threshold=angle_threshold,
            tolerance_widening=tolerance_widening
        )

        result = []
        result.append(f"Symmetry-reduced grain2: {np.round(sym_euler2[0], 4)} deg")
        result.append(f"Misorientation axis (approx.): {np.round(axis[0], 4)}")
        result.append(f"Misorientation angle (deg): {np.round(mis_ang, 4)}")

        if is_sig:
            sigma_axis_global = std_axis
            result.append(f"**This is a Σ{sigma_value} boundary**")
            result.append(f"Deviation from exact Σ boundary angle: {np.round(deviation, 4)} deg")
            result.append(f"Standard axis for Σ{sigma_value}: {std_axis}")
            result.append(f"Standard angle for Σ{sigma_value}: {predefined_angle} deg")
            std_euler_angles = crystal_rotation(euler1, std_axis, predefined_angle)
            dev_axis, dev_angle = misorientation_calc(sym_euler2, std_euler_angles)
            result.append(f"Additional deviation from exact orientation: {np.round(dev_angle,4)} deg, deviation axis: {np.round(dev_axis[0],4)}.")
        else:
            sigma_axis_global = None
            result.append("This is not a recognized low Σ boundary.")

        # Update both cubes
        draw_cube(canvas_g1, euler1, sigma_axis=sigma_axis_global)
        draw_cube(canvas_g2, euler2, sigma_axis=sigma_axis_global)

        # Update text
        output_text.delete('1.0', tk.END)
        output_text.insert(tk.END, "\n".join(result))

    except Exception as e:
        sigma_axis_global = None
        output_text.delete('1.0', tk.END)
        output_text.insert(tk.END, f"Error: {str(e)}")




def rotate_grain(grain_id, axis_vec, axis_index, direction, is_linked_call=False):
    global sigma_axis_global
    try:
        step = float(entry_step.get())
        angle = step * direction

        entry = entry_g1 if grain_id == 1 else entry_g2
        label = updated_g1 if grain_id == 1 else updated_g2
        canvas = canvas_g1 if grain_id == 1 else canvas_g2
        state = rotation_state_g1 if grain_id == 1 else rotation_state_g2

        # Get original (unrotated) Euler angles
        base_euler = list(map(float, entry.get().strip().split(',')))
        phi1, Phi, phi2 = np.deg2rad(base_euler)
        c1, s1 = np.cos(phi1), np.sin(phi1)
        c,  s  = np.cos(Phi),  np.sin(Phi)
        c2, s2 = np.cos(phi2), np.sin(phi2)
        Rz1 = np.array([[ c1,-s1,0],[ s1, c1,0],[  0,   0,1]])
        Rx  = np.array([[  1,   0,   0],[  0,   c,  -s],[  0,   s,   c]])
        Rz2 = np.array([[ c2,-s2,0],[ s2, c2,0],[  0,   0,   1]])
        R_base = Rz1 @ Rx @ Rz2

        # Update and apply global rotation
        if axis_index != -1:  # standard axis rotation, track cumulative
            state['rotations'][axis_index] += angle
            R_total = np.eye(3)
            for i, ax in enumerate([[1,0,0],[0,1,0],[0,0,1]]):
                if state['rotations'][i] != 0:
                    R_total = axis_angle_to_rotation_matrix(ax, state['rotations'][i]) @ R_total
            R_final = R_total @ R_base
        else:  # custom axis, apply accumulated rotation
            total_angle = state['custom']
            R_custom = axis_angle_to_rotation_matrix(axis_vec, total_angle)
            R_final = R_custom @ R_base

        def rotmat_to_euler(R):
            Phi = np.arccos(R[2,2])
            if abs(Phi) < 1e-6:
                phi1 = np.arctan2(-R[0,1], R[0,0])
                phi2 = 0
            elif abs(Phi - np.pi) < 1e-6:
                phi1 = np.arctan2(R[0,1], -R[0,0])
                phi2 = 0
            else:
                phi1 = np.arctan2(R[0,2], -R[1,2])
                phi2 = np.arctan2(R[2,0], R[2,1])
            return np.degrees([phi1 % (2*np.pi), Phi, phi2 % (2*np.pi)])

        new_euler = rotmat_to_euler(R_final)
        label.config(text=",".join([f"{x:.2f}" for x in new_euler]))



        # Draw with possible Σ axis
        draw_cube(canvas, new_euler, sigma_axis=sigma_axis_global)
        #Update global sigma axis if both orientations are ready
        update_sigma_axis_if_possible()
        def fmt_rot(rlist): return ", ".join([f"{r:.1f}°" for r in rlist])
        if grain_id == 1:
            rot_display_g1.config(text=fmt_rot(state['rotations']))
        else:
            rot_display_g2.config(text=fmt_rot(state['rotations']))

        if link_var.get() and not is_linked_call:
            other_id = 2 if grain_id == 1 else 1
            rotate_grain(other_id, axis_vec, axis_index, direction, is_linked_call=True)

    except Exception as e:
        messagebox.showerror("Error", str(e))

def rotate_custom_axis(grain_id, direction):
    try:
        step = float(entry_step.get())
        angle = step * direction

        axis_input = entry_custom_axis.get().strip()
        axis = list(map(float, axis_input.split(',')))
        if len(axis) != 3:
            raise ValueError("Custom axis must have 3 values, e.g., 1,0,0")
        norm = np.linalg.norm(axis)
        if norm == 0:
            raise ValueError("Axis vector cannot be zero.")
        axis = np.array(axis) / norm

        state = rotation_state_g1 if grain_id == 1 else rotation_state_g2
        state['custom'] += angle

        # Call the general rotation function (apply total angle)
        rotate_grain(grain_id, axis, axis_index=-1, direction=direction)
        display = custom_rot_g1 if grain_id == 1 else custom_rot_g2
        display.config(text=f"{state['custom']:.1f}°")
        
        # Handle link
        if link_var.get():
            other_id = 2 if grain_id == 1 else 1
            other_state = rotation_state_g2 if grain_id == 1 else rotation_state_g1
            other_state['custom'] += angle
            rotate_grain(other_id, axis, axis_index=-1, direction=0)
            other_display = custom_rot_g2 if grain_id == 1 else custom_rot_g1
            other_display.config(text=f"{other_state['custom']:.1f}°")
        global sigma_axis_global
        sigma_axis_global = None

    except Exception as e:
        messagebox.showerror("Invalid axis", str(e))


link_var = tk.BooleanVar(value=True)
check_link = ttk.Checkbutton(frame, text="Link G1 & G2 rotation", variable=link_var)
check_link.grid(row=11, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)

#  rotation panel on the right
rotation_panel = ttk.Frame(root, padding="10")
rotation_panel.grid(row=0, column=1, rowspan=3, sticky=(tk.N, tk.S))
axes = {'x': ([1, 0, 0], 0), 'y': ([0, 1, 0], 1), 'z': ([0, 0, 1], 2)}
for i, (label, (vec, idx)) in enumerate(axes.items()):
    ttk.Button(rotation_panel, text=f"+{label}",
               command=lambda v=vec, ix=idx: rotate_grain(1, v, ix, +1)).grid(row=i+1, column=0, padx=2, pady=2)
    ttk.Button(rotation_panel, text=f"-{label}",
               command=lambda v=vec, ix=idx: rotate_grain(1, v, ix, -1)).grid(row=i+1, column=1, padx=2, pady=2)

    ttk.Button(rotation_panel, text=f"+{label}",
               command=lambda v=vec, ix=idx: rotate_grain(2, v, ix, +1)).grid(row=i+1, column=2, padx=2, pady=2)
    ttk.Button(rotation_panel, text=f"-{label}",
               command=lambda v=vec, ix=idx: rotate_grain(2, v, ix, -1)).grid(row=i+1, column=3, padx=2, pady=2)
ttk.Button(rotation_panel, text="Reset G1", command=lambda: reset_rotation(1)).grid(row=5, column=0, columnspan=2, pady=(10, 2))
ttk.Button(rotation_panel, text="Reset G2", command=lambda: reset_rotation(2)).grid(row=5, column=2, columnspan=2, pady=(10, 2))
# Custom axis label and entry
ttk.Label(rotation_panel, text="Custom axis (x,y,z):").grid(row=7, column=0, columnspan=4, pady=(15, 2), sticky=tk.W)
entry_custom_axis = ttk.Entry(rotation_panel, width=20)
entry_custom_axis.insert(0, "1,1,1")
entry_custom_axis.grid(row=7, column=0, columnspan=4, padx=5, pady=2)
ttk.Button(rotation_panel, text="Rotate G1 +Axis",
           command=lambda: rotate_custom_axis(1, +1)).grid(row=9, column=0, columnspan=2, pady=2)
ttk.Button(rotation_panel, text="Rotate G1 -Axis",
           command=lambda: rotate_custom_axis(1, -1)).grid(row=10, column=0, columnspan=2, pady=2)

ttk.Button(rotation_panel, text="Rotate G2 +Axis",
           command=lambda: rotate_custom_axis(2, +1)).grid(row=9, column=2, columnspan=2, pady=2)
ttk.Button(rotation_panel, text="Rotate G2 -Axis",
           command=lambda: rotate_custom_axis(2, -1)).grid(row=10, column=2, columnspan=2, pady=2)

# Custom axis angle display
ttk.Label(rotation_panel, text="G1 Custom Axis Rotation:").grid(row=11, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
custom_rot_g1 = ttk.Label(rotation_panel, text="0.0°")
custom_rot_g1.grid(row=12, column=0, columnspan=2, sticky=tk.W)

ttk.Label(rotation_panel, text="G2 Custom Axis Rotation:").grid(row=11, column=2, columnspan=2, sticky=tk.W, pady=(10, 0))
custom_rot_g2 = ttk.Label(rotation_panel, text="0.0°")
custom_rot_g2.grid(row=12, column=2, columnspan=2, sticky=tk.W)


# Link checkbox
ttk.Checkbutton(rotation_panel, text="Link G1 & G2", variable=link_var).grid(row=6, column=0, columnspan=4, pady=(10, 2), sticky=tk.W)

# Separator label
ttk.Label(rotation_panel, text="Σ Boundary Result:").grid(row=13, column=0, columnspan=4, pady=(15, 2), sticky=tk.W)

# Output text box (new location)
output_text = tk.Text(rotation_panel, width=50, height=16, wrap=tk.WORD)
output_text.grid(row=14, column=0, columnspan=4, padx=5, pady=5)


# G1 rotation display
label_rot_g1 = ttk.Label(frame, text="G1 Rotation (X,Y,Z):")
label_rot_g1.grid(row=9, column=0, padx=5, pady=5, sticky=tk.W)
rot_display_g1 = ttk.Label(frame, text="0°, 0°, 0°")
rot_display_g1.grid(row=9, column=1, padx=5, pady=5, sticky=tk.W)

# G2 rotation display
label_rot_g2 = ttk.Label(frame, text="G2 Rotation (X,Y,Z):")
label_rot_g2.grid(row=10, column=0, padx=5, pady=5, sticky=tk.W)
rot_display_g2 = ttk.Label(frame, text="0°, 0°, 0°")
rot_display_g2.grid(row=10, column=1, padx=5, pady=5, sticky=tk.W)


root.mainloop()
