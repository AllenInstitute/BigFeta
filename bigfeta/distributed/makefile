ifeq ($(MAKECMDGOALS),allen)
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
endif

allen: src/bigfeta_dist_solve.o chkopts
	@echo ${CLINKER}
	${CLINKER} -o bin/bigfeta_dist_solve src/bigfeta_dist_solve.o  ${PETSC_KSP_LIB}
	${RM} src/bigfeta_dist_solve.o

cori:
	module load cray-petsc-64; \
        module load cray-hdf5-parallel; \
	cc -o ${PWD}/bin/bigfeta_solver_cori ${PWD}/src/bigfeta_dist_solve.c 
