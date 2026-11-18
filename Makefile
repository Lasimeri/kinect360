# kinect360/Makefile: out/kinect3d with warnings as errors; `make check` also gates every GLSL stage
# through glslangValidator (any warning fails) and runs the hardware selftest.
CC      = gcc
CFLAGS  = -std=gnu11 -O2 -Wall -Wextra -Wshadow -Wformat=2 -Wundef -Wstrict-prototypes -Werror \
          -isystem /usr/include/libfreenect
LDLIBS  = -lfreenect -lglfw -lOpenGL -lturbojpeg -lm -pthread
SHADERS = mesh_a.vert mesh_a.geom mesh_b.vert mesh.frag scatter.comp check.comp pip.vert pip.frag winner.comp pmap.comp wz.comp dfill.comp syn.frag comp.comp gn.comp sflow.comp cells.comp arrow.vert arrow.frag fg.comp \
          fl_gray.comp fl_pyr.comp fl_scharr.comp fl_lk.comp fl_fb.comp

SRCS    = src/kinect3d.c src/webcam.c src/flow.c src/fsrflow.c
FSR     = third_party/fsr3of
FSR_OBJ = $(FSR)/out/native/kern/framegen.o $(FSR)/out/native/kern/fsr3.o $(FSR)/out/native/kern/deblock.o \
          $(FSR)/out/native/kern/amdfi.o $(FSR)/out/native/kern/img_ref.o $(FSR)/out/native/kern/img_vx.o \
          $(FSR)/out/native/kern/bgra.o $(FSR)/out/native/kern/flow.o $(FSR)/out/native/engine/cpus.o \
          $(FSR)/out/native/engine/team.o
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Werror -I/usr/include/opencv5
FACELIBS = -lopencv_core -lopencv_imgproc -lopencv_objdetect -lopencv_dnn
$(FSR_OBJ):
	$(MAKE) -C $(FSR) $(FSR_OBJ:$(FSR)/%=%)

# The user's VisionTrack tracker (github Lasimeri/Intel-Phi-VisionTrack ac165e8, unmodified): its kernels with its
# own host flags (lane model, -ffp-contract=off: the card's bit-exact twin), one partial link, and every symbol but
# tk_* made local: its fg_*, fsr_*, img_* would otherwise collide with the FSR3 port's 44 same names.
# -O3 -march=native: bit-identical to its own -O2 build (vt_replay, 12 scenes x 300 frames 640x480: 0 of 3600 result
# hashes differ, 2026-11-18), 6.40x faster (128,749 -> 20,114 ms).
VT      = third_party/visiontrack
VT_KERN = img_ref img_vx lk_ref lk_vx detect framegen fsr3 stereo kalman track
VTFLAGS = -std=gnu11 -O3 -march=native -Wall -Wextra -Werror -ffp-contract=off -fno-math-errno -I$(VT)/src/common -I$(VT)/src/kern -I$(VT)/src/card -I$(VT)/src/host
out/vt/%.o: $(VT)/src/kern/%.c Makefile
	@mkdir -p out/vt
	$(CC) $(VTFLAGS) -c -o $@ $<
out/vt/vt_kern.o: $(VT_KERN:%=out/vt/%.o)
	ld -r -o out/vt/vt_all.o $^
	nm -g --defined-only out/vt/vt_all.o | awk '$$3 ~ /^tk_/ {print $$3}' > out/vt/api.syms
	@test -s out/vt/api.syms || { echo "no tk_ symbols"; exit 1; }
	objcopy --keep-global-symbols=out/vt/api.syms out/vt/vt_all.o $@
	@u=$$(nm -u $@ | awk '{print $$2}' | grep -v -x -E 'fmaf|fmaxf|fminf|fprintf|fputc|fwrite|_GLOBAL_OFFSET_TABLE_|memcpy|memset|__stack_chk_fail|stderr'); \
	 test -z "$$u" || { echo "vt_kern.o: unexpected undefined: $$u"; exit 1; }
out/obj/vtface.o: src/vtface.c src/vtface.h src/facetrack.h Makefile
	@mkdir -p out/obj
	$(CC) $(VTFLAGS) -Wshadow -Wstrict-prototypes -Isrc -c -o $@ $<

out/kinect3d: $(SRCS) $(FSR_OBJ) out/vt/vt_kern.o out/obj/vtface.o src/facetrack.cpp src/tribuf.h src/webcam.h src/flow.h src/facetrack.h src/fsrflow.h src/vtface.h Makefile
	@mkdir -p out/obj
	g++ $(CXXFLAGS) -c -o out/obj/facetrack.o src/facetrack.cpp
	for s in $(SRCS); do $(CC) $(CFLAGS) -isystem $(FSR)/src/engine -isystem $(FSR)/src/kern -isystem $(FSR)/src/common -c -o out/obj/$$(basename $$s .c).o $$s || exit 1; done
	g++ -o $@ out/obj/kinect3d.o out/obj/webcam.o out/obj/flow.o out/obj/fsrflow.o out/obj/facetrack.o out/obj/vtface.o out/vt/vt_kern.o $(FSR_OBJ) $(LDLIBS) $(FACELIBS)

shaders: out/kinect3d
	@mkdir -p out/shaders
	out/kinect3d --dump-shaders out/shaders
	@cd out/shaders && for s in $(SHADERS); do \
		r=$$(glslangValidator $$s 2>&1); rc=$$?; \
		if [ $$rc -ne 0 ] || printf '%s' "$$r" | grep -q -i -E 'warning|error'; then printf '%s\n' "$$r"; exit 1; fi; \
		echo "glslangValidator OK: $$s"; \
	done

check: shaders
	out/kinect3d --selftest

.PHONY: shaders check

OPENCV  = -I/usr/include/opencv5
out/kcalib: tools/kcalib.cpp Makefile
	g++ -O2 -std=c++17 -Wall -Wextra -Werror $(OPENCV) -o $@ $< -lopencv_core -lopencv_imgcodecs -lopencv_imgproc \
		-lopencv_features -lopencv_flann -lopencv_geometry -lopencv_calib

out/flowcheck: tools/flowcheck.cpp Makefile
	g++ -O2 -std=c++17 -Wall -Wextra -Werror $(OPENCV) -o $@ $< -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_video

# webcam <-> Kinect calibration: 5 snapshots from the live cameras, then the solver; writes calib/webcam.txt
calibrate: out/kinect3d out/kcalib
	@D=out/calib-$$(date +%Y%m%d-%H%M%S); mkdir -p $$D calib; \
	out/kinect3d --calib-dump $$D && out/kcalib $$D calib/webcam.txt

.PHONY: calibrate

# calibration from one moving light (phone flashlight) seen by both cameras; writes calib/webcam.txt
calibrate-track: out/kinect3d out/kcalib
	@D=out/track-$$(date +%Y%m%d-%H%M%S); mkdir -p $$D calib; \
	out/kinect3d --calib-track $$D 30 && out/kcalib $$D calib/webcam.txt

.PHONY: calibrate-track

# markerless calibration from the face (YuNet eyes, nose tip, mouth corners in both cameras); writes calib/webcam.txt
calibrate-face: out/kinect3d out/kcalib
	@D=out/face-$$(date +%Y%m%d-%H%M%S); mkdir -p $$D calib; \
	out/kinect3d --calib-face $$D 30 && out/kcalib $$D calib/webcam.txt --focal 690.75

.PHONY: calibrate-face

# ground truth for the velocity's motion vectors (EMA vs VisionTrack's LK refinement) on a real frame moved by known sub-pixel steps
out/vectest: tools/vectest.c src/fsrflow.c src/fsrflow.h $(FSR_OBJ) Makefile
	$(CC) $(CFLAGS) -isystem $(FSR)/src/engine -isystem $(FSR)/src/kern -isystem $(FSR)/src/common -o $@ tools/vectest.c src/fsrflow.c $(FSR_OBJ) -lm -pthread

# the alignment's evidence tools (tcc, warnings as errors): alscape = background NCC of poses and its 2D shift
# landscape on a frame pair from kinect3d --dump-align; posediff = how far calib files disagree, by depth
out/alscape: tools/alscape.c
	tcc -Wall -Werror -o $@ $< -lm
out/posediff: tools/posediff.c
	tcc -Wall -Werror -o $@ $< -lm

# compile_commands.json for clangd: every source with the flags it is built with (absolute paths, so not published)
CDB_FSR = $(CC) $(CFLAGS) -isystem $(FSR)/src/engine -isystem $(FSR)/src/kern -isystem $(FSR)/src/common
CDB_TCC = cc -std=gnu11 -Wall -Wextra
compdb:
	@{ printf '['; s=''; \
	  e() { printf '%s\n {"directory": "%s", "file": "%s", "command": "%s -c %s"}' "$$s" "$(CURDIR)" "$$2" "$$1" "$$2"; s=','; }; \
	  for f in src/kinect3d.c src/webcam.c src/flow.c src/fsrflow.c tools/vectest.c; do e "$(CDB_FSR)" $$f; done; \
	  e "$(CC) $(VTFLAGS) -Wshadow -Wstrict-prototypes -Isrc" src/vtface.c; \
	  e "g++ $(CXXFLAGS)" src/facetrack.cpp; \
	  for f in tools/kcalib.cpp tools/flowcheck.cpp tools/facetest.cpp; do e "g++ -std=c++17 -Wall -Wextra $(OPENCV)" $$f; done; \
	  for f in tools/alscape.c tools/posediff.c tools/boxlink.c src/kinect-probe.c src/kinect-streams.c src/v4l2-probe.c src/holes.c src/shiftsearch.c; do e "$(CDB_TCC)" $$f; done; \
	  printf '\n]\n'; } > compile_commands.json
	@echo "compile_commands.json: $$(grep -c '"file"' compile_commands.json) entries"

.PHONY: compdb
