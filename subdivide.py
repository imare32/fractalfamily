"""
Subdivide a Bezier curve keeping original Shape.
see https://stackoverflow.com/a/879213
"""
from dataclasses import dataclass
from itertools import chain

import bpy
from mathutils import Vector


@dataclass
class Segment:
    start: Vector
    ctrl1: Vector
    ctrl2: Vector
    end: Vector

    def partial_seg(self, t0=0.0, t1=1.0):
        """
        Calculate a partial segment of the Bezier curve from t0 to t1.

        Args:
            t0 (float): Start parameter (0.0 to 1.0).
            t1 (float): End parameter (0.0 to 1.0).

        Returns:
            Segment: A new Segment representing the curve portion.
        """
        p0, p1, p2, p3 = self.start, self.ctrl1, self.ctrl2, self.end

        if t0 > t1:
            t0, t1 = t1, t0

        # Let's make at least the line segments of predictable length
        if p0 == p1 and p2 == p3:
            pt0 = p0 * (1 - t0) + p2 * t0
            pt1 = p0 * (1 - t1) + p2 * t1
            return Segment(pt0, pt0, pt1, pt1)

        u0 = 1.0 - t0
        u1 = 1.0 - t1

        qa = p0 * (u0 * u0) + p1 * (2 * t0 * u0) + p2 * (t0 * t0)
        qb = p0 * (u1 * u1) + p1 * (2 * t1 * u1) + p2 * (t1 * t1)
        qc = p1 * (u0 * u0) + p2 * (2 * t0 * u0) + p3 * (t0 * t0)
        qd = p1 * (u1 * u1) + p2 * (2 * t1 * u1) + p3 * (t1 * t1)

        pta = qa * u0 + qc * t0
        ptb = qa * u1 + qc * t1
        ptc = qb * u0 + qd * t0
        ptd = qb * u1 + qd * t1

        return Segment(pta, ptb, ptc, ptd)


def get_spline_segs(spline: bpy.types.Spline):
    """
    Extract segments from a Blender spline.

    Args:
        spline (bpy.types.Spline): The source spline.

    Returns:
        list[Segment]: List of Segment objects derived from the spline.
    """
    pts = spline.bezier_points
    segs = [Segment(pts[i - 1].co, pts[i - 1].handle_right, pts[i].handle_left, pts[i].co) for i in range(1, len(pts))]
    if spline.use_cyclic_u:
        segs.append(Segment(pts[-1].co, pts[-1].handle_right, pts[0].handle_left, pts[0].co))
    return segs


def subdivide_seg(orig_seg: Segment, no_segs=1):
    """
    Subdivide a single segment into multiple smaller segments.

    Args:
        orig_seg (Segment): The original segment.
        no_segs (int): Number of subdivisions.

    Returns:
        list[Segment]: List of subdivided segments.
    """
    if no_segs < 2:
        return [orig_seg]

    segs = []
    old_t = 0.0

    for i in range(0, no_segs - 1):
        t = (i + 1) / no_segs
        seg = orig_seg.partial_seg(old_t, t)
        segs.append(seg)
        old_t = t

    seg = orig_seg.partial_seg(old_t, 1.0)
    segs.append(seg)

    return segs


def subdivide_curve(curve: bpy.types.Curve, no_segs=1):
    """
    Subdivide all segments of a curve.

    Args:
        curve (bpy.types.Curve): The curve object to modify.
        no_segs (int): Number of subdivisions per segment.
    """
    orig_spline = curve.splines[0]
    is_cyclic = orig_spline.use_cyclic_u
    segs = get_spline_segs(orig_spline)
    segs = list(chain.from_iterable(subdivide_seg(seg, no_segs) for seg in segs))

    bezier_pts_info: list[list[Vector]] = []

    prev_seg = None
    for i, seg in enumerate(segs):
        pt = seg.start
        handle_right = seg.ctrl1

        if i == 0:
            if is_cyclic:
                handle_left = segs[-1].ctrl2
            else:
                handle_left = pt
        else:
            handle_left = prev_seg.ctrl2

        bezier_pts_info.append([pt, handle_left, handle_right])
        prev_seg = seg

    if is_cyclic:
        bezier_pts_info[-1][2] = segs[-1].ctrl1
    else:
        bezier_pts_info.append([prev_seg.end, prev_seg.ctrl2, prev_seg.end])

    spline = curve.splines.new('BEZIER')
    spline.use_cyclic_u = is_cyclic
    spline.bezier_points.add(len(bezier_pts_info) - 1)

    for i, new_point in enumerate(bezier_pts_info):
        spline.bezier_points[i].co = new_point[0]
        spline.bezier_points[i].handle_left = new_point[1]
        spline.bezier_points[i].handle_right = new_point[2]
        spline.bezier_points[i].handle_right_type = 'FREE'

    curve.splines.remove(orig_spline)
