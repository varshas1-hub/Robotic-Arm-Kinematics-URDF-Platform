import math
import time
import xml.etree.ElementTree as ET
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# FastAPI Application Instance
app = FastAPI(title="Robotic Arm Kinematics & Speed Verification Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

research_benchmarks = []

class LinkConfig(BaseModel):
    length: float
    yaw: float = 0.0
    pitch: float = 0.0

class TrajectoryRequest(BaseModel):
    shape: str
    links: List[LinkConfig]
    motor_speed_dps: float = 90.0
    num_samples: int = 140

class MatrixSweepRequest(BaseModel):
    shape: str = "star"
    link_configurations: List[List[float]]
    motor_speeds: List[float]

def dist3d(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2])

# Multi-Link Forward Reaching Kinematic Solver with Elbow-Up Arc Constraint
def solve_arm_angles(target_x: float, target_y: float, target_z: float, lengths: List[float], bx=0.0, by=0.3, bz=0.0):
    dx = target_x - bx
    dy = target_y - by
    dz = target_z - bz

    # Base yaw to face the target directly
    base_yaw = math.atan2(dx, dz)
    horizontal_dist = math.hypot(dx, dz)
    direct_dist = math.hypot(horizontal_dist, dy)
    
    total_reach = sum(lengths)
    is_reachable = True
    
    # Envelope safety bounds
    if direct_dist > (total_reach * 0.98):
        is_reachable = False
        ratio = (total_reach * 0.98) / max(direct_dist, 0.001)
        horizontal_dist *= ratio
        dy *= ratio
        direct_dist = total_reach * 0.98
    elif direct_dist < (total_reach * 0.15):
        ratio = (total_reach * 0.15) / max(direct_dist, 0.001)
        horizontal_dist *= ratio
        dy *= ratio
        direct_dist = total_reach * 0.15

    target_2d = [horizontal_dist, dy]
    n = len(lengths)

    # Seed with an industrial elbow-up arc
    points_2d = [[0.0, 0.0]]
    cum_len = 0.0
    for l in lengths:
        cum_len += l
        t_norm = cum_len / total_reach
        rx = target_2d[0] * t_norm
        arch = math.sin(t_norm * math.pi) * (total_reach * 0.4)
        ry = target_2d[1] * t_norm + arch
        points_2d.append([rx, ry])

    # FABRIK iterations
    for _ in range(18):
        # Backward Pass (target tip inward)
        points_2d[-1] = [target_2d[0], target_2d[1]]
        for i in range(n - 1, -1, -1):
            p_next = points_2d[i + 1]
            p_curr = points_2d[i]
            d = math.hypot(p_next[0] - p_curr[0], p_next[1] - p_curr[1])
            if d > 1e-6:
                r = lengths[i] / d
                points_2d[i] = [
                    p_next[0] + (p_curr[0] - p_next[0]) * r,
                    p_next[1] + (p_curr[1] - p_next[1]) * r
                ]

        # Forward Pass (base outward)
        points_2d[0] = [0.0, 0.0]
        for i in range(n):
            p_curr = points_2d[i]
            p_next = points_2d[i + 1]
            d = math.hypot(p_next[0] - p_curr[0], p_next[1] - p_curr[1])
            if d > 1e-6:
                r = lengths[i] / d
                nx = p_curr[0] + (p_next[0] - p_curr[0]) * r
                ny = p_curr[1] + (p_next[1] - p_curr[1]) * r
                nx = max(0.0, nx)
                if i < n - 1:
                    ny = max(dy, ny)
                points_2d[i + 1] = [nx, ny]

    # Convert 2D reach back into 3D world space
    joints_3d = []
    sin_yaw, cos_yaw = math.sin(base_yaw), math.cos(base_yaw)
    for p in points_2d:
        joints_3d.append({
            "x": round(bx + p[0] * sin_yaw, 3),
            "y": round(by + p[1], 3),
            "z": round(bz + p[0] * cos_yaw, 3)
        })

    q_shoulder = math.atan2(points_2d[1][1], max(points_2d[1][0], 1e-4))
    q_elbow = math.atan2(points_2d[2][1] - points_2d[1][1], max(points_2d[2][0] - points_2d[1][0], 1e-4)) - q_shoulder if n > 1 else 0.0

    angles = {
        "joint_waist": round(base_yaw, 4),
        "joint_shoulder": round(q_shoulder, 4),
        "joint_elbow": round(q_elbow, 4),
        "joint_wrist": 0.0
    }

    can_draw = is_reachable and (abs(joints_3d[-1]["y"] - target_y) < 0.35)
    return joints_3d, angles, can_draw, is_reachable

def build_urdf_xml(lengths: List[float]):
    xml = '<?xml version="1.0"?>\n<robot name="custom_speed_arm">\n'
    xml += '  <link name="base_link">\n'
    xml += '    <visual><geometry><cylinder length="0.3" radius="1.1"/></geometry></visual>\n'
    xml += '  </link>\n'
    xml += '  <joint name="joint_waist" type="revolute">\n'
    xml += '    <parent link="base_link"/><child link="link_1"/>\n'
    xml += '    <origin xyz="0 0 0.15" rpy="0 0 0"/><axis xyz="0 1 0"/>\n'
    xml += '    <limit lower="-3.14" upper="3.14" effort="10.0" velocity="3.14"/>\n'
    xml += '  </joint>\n'

    for i, l in enumerate(lengths):
        link_name = f"link_{i+1}"
        xml += f'  <link name="{link_name}">\n'
        xml += f'    <visual><origin xyz="0 {l/2:.3f} 0"/><geometry><cylinder length="{l:.3f}" radius="0.14"/></geometry></visual>\n'
        xml += f'  </link>\n'

        if i < len(lengths) - 1:
            j_name = f"joint_{i+1}"
            child_name = f"link_{i+2}"
            xml += f'  <joint name="{j_name}" type="revolute">\n'
            xml += f'    <parent link="{link_name}"/><child link="{child_name}"/>\n'
            xml += f'    <origin xyz="0 {l:.3f} 0"/><axis xyz="1 0 0"/>\n'
            xml += f'    <limit lower="-2.5" upper="2.5" effort="10.0" velocity="3.14"/>\n'
            xml += '  </joint>\n'

    xml += '</robot>'
    return xml

# URDF Parser Endpoint
@app.post("/api/arm/parse-urdf")
async def parse_user_urdf(file: UploadFile = File(...)):
    try:
        content = await file.read()
        root = ET.fromstring(content.decode("utf-8"))
        robot_name = root.attrib.get("name", "User Arm")
        extracted_links = []
        joints_found = []

        for link in root.findall(".//link"):
            link_name = link.attrib.get("name", "")
            if "base" in link_name.lower():
                continue
            cyl = link.find(".//geometry/cylinder")
            if cyl is not None and "length" in cyl.attrib:
                extracted_links.append(float(cyl.attrib["length"]))

        for joint in root.findall(".//joint"):
            j_name = joint.attrib.get("name", "unnamed")
            j_type = joint.attrib.get("type", "unknown")
            joints_found.append({"name": j_name, "type": j_type})

        if not extracted_links:
            extracted_links = [3.0, 2.5, 1.5]

        total_reach = sum(extracted_links)

        return {
            "status": "success",
            "robot_name": robot_name,
            "link_count": len(extracted_links),
            "link_lengths": extracted_links,
            "total_reach": round(total_reach, 2),
            "joints": joints_found,
            "urdf_xml": content.decode("utf-8")
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": f"Invalid URDF XML: {str(e)}"})

# Trajectory Calculation Endpoint
@app.post("/api/arm/trajectory")
def generate_trajectory(req: TrajectoryRequest):
    shape = req.shape
    n_samples = req.num_samples
    lengths = [l.length for l in req.links]
    total_reach = sum(lengths)

    table_y = 0.02
    scale = min(total_reach * 0.45, 5.0)

    path_points = []
    frames = []

    for i in range(n_samples + 1):
        t = (i / n_samples) * 2 * math.pi

        if shape == "circle":
            tx, tz, ty = scale * math.cos(t), scale * math.sin(t), table_y
        elif shape == "square":
            side = (i / n_samples) * 4.0
            half = scale * 0.75
            if side < 1.0: tx, tz = -half + side * (2 * half), -half
            elif side < 2.0: tx, tz = half, -half + (side - 1.0) * (2 * half)
            elif side < 3.0: tx, tz = half - (side - 2.0) * (2 * half), half
            else: tx, tz = -half, half - (side - 3.0) * (2 * half)
            ty = table_y
        elif shape == "star":
            num_points = 10
            angle_step = (2 * math.pi) / num_points
            seg = int(t / angle_step)
            frac = (t % angle_step) / angle_step
            r1 = scale if (seg % 2 == 0) else (scale * 0.45)
            r2 = (scale * 0.45) if (seg % 2 == 0) else scale
            a1 = seg * angle_step - math.pi / 2
            a2 = (seg + 1) * angle_step - math.pi / 2
            tx = r1 * math.cos(a1) + frac * (r2 * math.cos(a2) - r1 * math.cos(a1))
            tz = r1 * math.sin(a1) + frac * (r2 * math.sin(a2) - r1 * math.sin(a1))
            ty = table_y
        elif shape == "helix":
            tx = (scale * 0.7) * math.cos(t * 3)
            tz = (scale * 0.7) * math.sin(t * 3)
            ty = table_y + (i / n_samples) * (total_reach * 0.4)
        elif shape == "lissajous":
            tx, tz, ty = scale * math.sin(3 * t), scale * math.cos(2 * t), table_y
        else:
            tx, tz, ty = scale, 0.0, table_y

        path_points.append((tx, ty, tz))
        joints, angles, can_draw, is_reachable = solve_arm_angles(tx, ty, tz, lengths, bx=0.0, by=0.3, bz=0.0)

        frames.append({
            "joints": joints,
            "angles": angles,
            "target": {"x": tx, "y": ty, "z": tz},
            "can_draw": can_draw,
            "is_reachable": is_reachable
        })

    arc_length = 0.0
    for k in range(1, len(path_points)):
        if frames[k]["can_draw"]:
            arc_length += dist3d(path_points[k - 1], path_points[k])

    theoretical_time = round((360.0 * len(lengths)) / max(req.motor_speed_dps, 1.0), 3)
    step_delay_ms = max(16, int((theoretical_time * 1000) / n_samples))

    return {
        "frames": frames,
        "arc_length": round(arc_length, 3),
        "step_delay_ms": step_delay_ms,
        "theoretical_time": theoretical_time,
        "urdf_xml": build_urdf_xml(lengths)
    }

# Multi-Config Speed Matrix Verification
@app.post("/api/arm/verify-matrix")
def verify_speed_matrix(req: MatrixSweepRequest):
    results = []
    for lengths in req.link_configurations:
        total_reach = sum(lengths)
        for spd in req.motor_speeds:
            traj_req = TrajectoryRequest(
                shape=req.shape,
                links=[LinkConfig(length=l) for l in lengths],
                motor_speed_dps=spd,
                num_samples=100
            )
            sim = generate_trajectory(traj_req)
            arc = sim["arc_length"]
            time_sec = sim["theoretical_time"]
            avg_linear = round(arc / max(time_sec, 0.001), 2)

            entry = {
                "shape": req.shape.upper(),
                "link_count": len(lengths),
                "link_lengths": "-".join(str(x) for x in lengths),
                "total_reach": round(total_reach, 2),
                "motor_speed_dps": spd,
                "arc_length": arc,
                "elapsed_time_sec": time_sec,
                "avg_linear_speed": avg_linear,
                "status": "NOMINAL" if arc > 0.5 else "STALLED"
            }
            results.append(entry)
            research_benchmarks.append(entry)

    return {"matrix_results": results, "total_runs": len(results)}

class LogEntry(BaseModel):
    test_id: str
    shape: str
    link_count: int
    link_lengths: str
    total_reach: float
    motor_speed_dps: float
    arc_length: float
    elapsed_time_sec: float
    avg_linear_speed: float
    avg_angular_speed: float
    status: str

@app.post("/api/arm/log")
def log_experiment(data: LogEntry):
    research_benchmarks.append(data.dict())
    return {"status": "recorded", "count": len(research_benchmarks)}

@app.get("/api/arm/benchmarks")
def get_benchmarks():
    return research_benchmarks

@app.delete("/api/arm/benchmarks")
def clear_benchmarks():
    research_benchmarks.clear()
    return {"status": "cleared"}

# Static Files Mount & Web Routing
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)