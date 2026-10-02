# The physics unit tests have not kept up with the source tree: their CMakeLists
# still lists pre-reorganization paths, and the sources themselves predate the
# templatization of the physics modules (e.g. NavierStokes.cpp constructs
# 'navierstokes' with no template arguments).  They do not build, so they are
# left out here rather than breaking the default target for everyone.  Re-enable
# once they have been brought up to date.
#ADD_SUBDIRECTORY(physics)

ADD_SUBDIRECTORY(checkpointing)
