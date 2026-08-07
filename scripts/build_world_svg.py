#!/usr/bin/env python3
"""Regenerate data/world_map_paths.svg from Natural Earth (public domain).

Source: Natural Earth 1:110m "land" vector, from the natural-earth-vector
repository (https://github.com/nvkelso/natural-earth-vector). Natural Earth is
released into the public domain: "no permission is needed to use Natural Earth.
Crediting the authors is unnecessary." (naturalearthdata.com/about/terms-of-use)

Regenerating from source gives the outline a documented, verifiable provenance
rather than an undocumented asset of unknown origin.

Projection must match core.mapview._xy exactly, or the culture points drift:
    x = (lon + 180) / 360 * W
    y = (LAT_TOP - lat) / (LAT_TOP - LAT_BOT) * H

Usage: python scripts/build_world_svg.py <ne_110m_land.geojson> [out.svg]
"""
import json, sys, os

W, H, LAT_TOP, LAT_BOT = 1000, 520, 85.0, -60.0   # identical to core/mapview.py


def xy(lon, lat):
    return ((lon + 180) / 360 * W, (LAT_TOP - lat) / (LAT_TOP - LAT_BOT) * H)


def ring_to_path(ring, min_area_px=1.5):
    # the canvas covers LAT_TOP..LAT_BOT only; skip rings entirely outside it
    lats = [lat for _lon, lat, *_ in ring]
    if max(lats) < LAT_BOT or min(lats) > LAT_TOP:
        return None
    # clamp the rest so nothing is drawn off-canvas (e.g. Antarctic fringes)
    pts = [xy(lon, min(LAT_TOP, max(LAT_BOT, lat))) for lon, lat, *_ in ring]
    # drop specks that would render as invisible dots
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    if (max(xs) - min(xs)) * (max(ys) - min(ys)) < min_area_px:
        return None
    out = [f"M{pts[0][0]:.1f} {pts[0][1]:.1f}"]
    last = pts[0]
    for x, y in pts[1:]:
        if abs(x - last[0]) < 0.15 and abs(y - last[1]) < 0.15:
            continue                      # thin out sub-pixel detail
        out.append(f"L{x:.1f} {y:.1f}")
        last = (x, y)
    return "".join(out) + "Z" if len(out) > 3 else None


def main():
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "world_map_paths.svg")
    gj = json.load(open(src, encoding="utf-8"))
    paths = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        polys = ([geom["coordinates"]] if geom["type"] == "Polygon"
                 else geom["coordinates"])
        for poly in polys:
            for ring in poly:                 # outer ring + holes, both drawn
                d = ring_to_path(ring)
                if d:
                    paths.append(f'<path d="{d}"/>')
    with open(out, "w", encoding="utf-8") as f:
        f.write("<!-- World land outline generated from Natural Earth 1:110m (public "
                "domain, naturalearthdata.com) by scripts/build_world_svg.py. "
                "Equirectangular projection matching core/mapview.py. -->\n")
        f.write("\n".join(paths) + "\n")
    print(f"wrote {out}: {len(paths)} paths, {os.path.getsize(out)/1024:.0f} KB")


if __name__ == "__main__":
    main()
