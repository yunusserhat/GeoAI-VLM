# Third-Party Notices

## Vision2Slope

GeoAI-VLM vendors modules and road slope estimation helpers from Vision2Slope (since v0.3):

- Repository: https://github.com/CubicsYang/Vision2Slope

Vision2Slope is distributed under the MIT License:

```text
MIT License

Copyright (c) 2025 CHEN YANG

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Runtime resources (not distributed)

No other third-party code is vendored. The optional research demo
(`geoai_vlm.app`) makes the viewer's browser load, at run time:

- Leaflet 1.9.4 from unpkg.com (BSD 2-Clause License, https://leafletjs.com),
- map tiles from tile.openstreetmap.org (data (c) OpenStreetMap contributors,
  ODbL; use is subject to the OpenStreetMap tile usage policy).

`examples/build_demo_index.py` reads part of the dataset `yunusserhat/fatih`
(CC BY-SA 4.0, doi:10.57967/hf/10144; imagery and image metadata from
Mapillary contributors, CC BY-SA 4.0). Nothing from it is included in this
repository or its packages; the attribution is written into the index it
builds.
