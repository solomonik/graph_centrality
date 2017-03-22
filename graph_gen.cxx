#include "graph_aux.h"
#include "generator/make_graph.h"

uint64_t gen_graph(int scale, int edgef, uint64_t seed, uint64_t **edges) {

  uint64_t nedges;
  double   initiator[4] = {.57, .19, .19, .05};
  CTF::Timer tmrg("gen_graph");
  tmrg.start();
  make_graph(scale, (((int64_t)1)<<scale)*edgef, seed, seed+1, initiator, (int64_t *)&nedges, (int64_t **)edges);
  tmrg.stop();

  return nedges;
}


uint64_t norm_graph(uint64_t *ed, uint64_t ned) {

	uint64_t l, n;

	if (ned == 0) return 0;

//	qsort(ed, ned, sizeof(uint64_t[2]), cmpedge);
	// record degrees considering multiple edges
	// and self-loop and remove them from edge list
	for(n = l = 1; n < ned; n++) {

		if (((ed[2*n]   != ed[2*(n-1)]  )  ||
		     (ed[2*n+1] != ed[2*(n-1)+1])) &&
		     (ed[2*n] != ed[2*n+1])) {

			ed[2*l]   = ed[2*n];
			ed[2*l+1] = ed[2*n+1];
			l++;
		}
	}
	return l;
}

