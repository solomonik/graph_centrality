/** \addtogroup examples 
  * @{ 
  * \defgroup btwn_central betweenness centrality
  * @{ 
  * \brief betweenness centrality computation
  */

#include <float.h>
#include "btwn_central.h"
#include "generator/make_graph.h"
#define __STDC_FORMAT_MACROS 1
#include <inttypes.h>

using namespace CTF;



void btwn_cnt_fast(Matrix<wht> A, int64_t b, Vector<real> & v, int nbatches=0, bool sp_B=true, bool sp_C=true, bool adapt=true){
  assert(sp_B || !sp_C);
  assert(!adapt || (sp_B && sp_C));
  World dw = *A.wrld;
  int64_t n = A.nrow;

  Semiring<mpath> mp  = get_mpath_semiring();
  Monoid<cmpath> mcmp = get_cmpath_monoid();
  Monoid<cpath>  mcp  = get_cpath_monoid();


  Matrix<mpath> speye(n,n,SP,dw,mp);
  Scalar<mpath> sm(mpath(0,1),dw,mp);
  speye["ii"] = sm[""];
  Transform<wht>([=](wht& w){ w = MAX_WHT; })(A["ii"]);
  Bivar_Function<wht,mpath,mpath> * Bellman = get_Bellman_kernel();
  Bivar_Function<wht,cpath,cmpath> * Brandes = get_Brandes_kernel();
  

  for (int64_t ib=0; ib<n && (nbatches == 0 || ib/b<nbatches); ib+=b){
    int64_t k = std::min(b, n-ib);

    //initialize shortest mpath vectors from the next k sources to the corresponding columns of the adjacency matrices and loops with weight 0
    //Transform<int>([=](int& w){ w = 0; })(A["ii"]);
    Tensor<wht> iA = A.slice(ib*n, (ib+k-1)*n+n-1);

    //let shortest mpaths vectors be mpaths
    int atr_C = 0;
    if (sp_C) atr_C = atr_C | SP;
    Matrix<mpath> B(n, k, atr_C, dw, mp, "B");
    Matrix<mpath> all_B(n, k, dw, mp, "all_B");
    B["ij"] = Function<wht,mpath>([](wht w){ return mpath(w, 1); })(iA["ij"]);


//    B.leave_home();
//    all_B.leave_home();
     
    //compute Bellman Ford
    int nbl = 0;
#ifndef TEST_SUITE
    double sbl = MPI_Wtime();
#endif
    all_B["ij"] = B["ij"]; 

    
    Scalar<int> num_init(dw); 
    num_init[""] += Function<mpath,int>([](mpath p){ return p.w<MAX_WHT; })(B["ij"]);
    int64_t nnz_last = num_init.get_val();
    double t_all_last = 0.0, t_bm_last = 0.0;
    int64_t nnz_out = 0;
    int last_type = 0;
    for (int i=0; i<n; i++, nbl++){
      Matrix<mpath> * dns_B = NULL;
      double t_st = MPI_Wtime();
      Matrix<mpath> C(B);
      B.set_zero();
//      C.leave_home();
      if (sp_B || sp_C){
        if (sp_B) C.sparsify([](mpath p){ return p.w < MAX_WHT; });
        if (dw.rank == 0 && i!= 0){
          if (!sp_C || last_type){
            nnz_out = C.nnz_tot;
            printf("Bellman (dns=1) [filtered_nnz_C = %ld] <- [nnz_A = %ld] * [nnz_B = %ld] took time %lf (%lf)\n",nnz_out,A.nnz_tot,nnz_last,t_bm_last,t_all_last);
         } else
            printf("Bellman (dns=0) [computed_nnz_C = %ld] <- [nnz_A = %ld] * [nnz_B = %ld] took time %lf (%lf)\n",nnz_out,A.nnz_tot,nnz_last,t_bm_last,t_all_last);
        }
        nnz_last = C.nnz_tot;
        if (C.nnz_tot == 0){ nbl--; break; }
      }
      CTF::Timer tbl("Bellman");
      tbl.start();
      double t_bm_st = MPI_Wtime();
      Matrix<mpath> * pB = &B;
      if (sp_C && adapt && (((double)A.nnz_tot)*C.nnz_tot)/n >= ((double)n)*k/4.){
        last_type = 1;
        dns_B = new Matrix<mpath>(n, k, dw, mp, "dns_B");
//        dns_B->leave_home();
        (*Bellman)(A["ik"],C["kj"],(*dns_B)["ij"]);
        pB = dns_B;
        //dns_B.sparsify();
        //B["ij"] += dns_B["ij"];
      } else {
        last_type = 0;
        (*Bellman)(A["ik"],C["kj"],B["ij"]);
      }
      double t_bm = MPI_Wtime() - t_bm_st;
      if (sp_C && last_type == 0) nnz_out = B.nnz_tot;
      tbl.stop();
      CTF::Timer tblp1("Bellman_post_tform1");
      tblp1.start();
      Transform<mpath,mpath>([](mpath p, mpath & q){ if (p.w<q.w || (p.w==q.w && q.m==0)) q.w = MAX_WHT; } )(all_B["ij"],(*pB)["ij"]);
      tblp1.stop();
      if (sp_C) pB->sparsify([](mpath p){ return p.w < MAX_WHT; });
      CTF::Timer tblp2("Bellman_post_tform2");
      tblp2.start();
      all_B["ij"] += (*pB)["ij"];
      //Transform<mpath,mpath>([](mpath p, mpath & q){ if (p.w <= q.w){ if (p.w < q.w){ q=p; } else if (p.m > 0){ q.m+=p.m; } } })((*pB)["ij"],all_B["ij"]); 
      tblp2.stop();
      double t_all = MPI_Wtime() - t_st;
      if (!sp_B && !sp_C){
        Scalar<int> num_changed(dw); 
        num_changed[""] += Function<mpath,int>([](mpath p){ return p.w<MAX_WHT; })(B["ij"]);
        int64_t nnz_new = num_changed.get_val();
        if (dw.rank == 0){
          printf("Bellman (dns=1) [filtered_nnz_C = %ld] <- [nnz_A = %ld] * [nnz_B = %ld] took time %lf (%lf)\n",nnz_new,A.nnz_tot,nnz_last,t_bm,t_all);
          nnz_last = nnz_new;
        }
        if (nnz_new == 0) break;
      } else {
        t_all_last = t_all;
        t_bm_last = t_bm;
      }
      if (dns_B != NULL){
        B = Matrix<mpath>(*dns_B);
        delete dns_B;
      }
    }
    Tensor<mpath> ispeye = speye.slice(ib*n, (ib+k-1)*n+n-1);
    all_B["ij"] += ispeye["ij"];
    
#ifndef TEST_SUITE
    double timbl = MPI_Wtime() - sbl;
#endif
#ifndef TEST_SUITE
    double sbr = MPI_Wtime();
#endif

    CTF::Timer tbr("Brandes");
    CTF::Timer tbrs("Brandes_post_spfy");
    CTF::Timer tbra("Brandes_post_add");
    CTF::Timer tbrp("Brandes_post_tform");

    Matrix<cmpath> all_cB(n, k, dw, mcmp, "all_cB");
//    all_cB.leave_home();

    int atr_B = 0;
    if (sp_B) atr_B = atr_B | SP;
    Matrix<cpath> C(n, k, atr_B, dw, mcp, "C");
//    C.leave_home();
    C["ij"] = Function<mpath,cpath>([](mpath p){ return cpath(p.w, 1./p.m); })(all_B["ij"]);
    all_cB["ij"] += Function<cpath,cmpath>([](cpath p){ return cmpath(p.w, -1, 0.0); })(C["ij"]);
    tbr.start();
    all_cB["ij"] += (*Brandes)(A["ki"],C["kj"]);
    tbr.stop();
    //compute centrality scores by propagating them backwards from the furthest nodes (reverse Bellman Ford)
    int nbr = 0;
    Matrix<cmpath> cB(all_cB);
//    cB.leave_home();
    //transfer shortest mpath data to Matrix of cmpaths to compute c centrality scores
    //Matrix<cmpath> cB(n, k, atr_C, dw, cp, "cB");
    Transform<mpath,cmpath>([](mpath p, cmpath & cp){ cp.c += 1./p.m;  })(all_B["ij"],cB["ij"]);
    if (sp_C)
      cB.sparsify([](cmpath p){ return p.m == -1. && p.w != 0.; });
    else 
      Transform<cmpath>([](cmpath & p){ if (p.m != -1 || p.w == 0.0) p = cmpath(-MAX_WHT,0,0); })(cB["ij"]);
    Transform<cmpath>([](cmpath & p){ p.c = 0.0; if (p.m == -1) p.m = 0; else p.m=-2-p.m; })(all_cB["ij"]);

    nnz_last = n*k-k;
    for (int i=0; i<n; i++, nbr++){
      Matrix<cmpath> * dns_cB = NULL;
      double t_st = MPI_Wtime();
      C.set_zero();
      C["ij"] += Function<cmpath,cpath>([](cmpath p){ return cpath(p.w, p.c); })(cB["ij"]);
      if (sp_B || sp_C){
        if (!sp_C || i==0) C.sparsify([](cpath p){ return p.w > 0 && p.w != MAX_WHT && p.c != 0.0; });
        if (!sp_C || last_type) nnz_out = C.nnz_tot;
        if (dw.rank == 0 && i!= 0){
          if (!sp_C || last_type)
            printf("Brandes (dns=1) [filtered_nnz_C = %ld] <- [nnz_A = %ld] * [nnz_B = %ld] took time %lf (%lf)\n",nnz_out,A.nnz_tot,nnz_last,t_bm_last,t_all_last);
          else
            printf("Brandes (dns=0) [computed_nnz_C = %ld] <- [nnz_A = %ld] * [nnz_B = %ld] took time %lf (%lf)\n",nnz_out,A.nnz_tot,nnz_last,t_bm_last,t_all_last);
        }
        nnz_last = C.nnz_tot;
        if (C.nnz_tot == 0){ nbr--; break; }
      }
      cB.set_zero();
      tbr.start();
      double t_bm_st = MPI_Wtime();
      Matrix<cmpath> * pcB = &cB;
      if (sp_C && adapt && (((double)A.nnz_tot)*C.nnz_tot)/n >= ((double)n)*k/4.){
        dns_cB = new Matrix<cmpath>(n, k, dw, mcmp, "dns_cB");
//        dns_cB->leave_home();
        (*dns_cB)["ij"] += (*Brandes)(A["ki"],C["kj"]);
        pcB = dns_cB;
        //dns_cB.sparsify();
        //cB["ij"] += dns_cB["ij"];
        last_type = 1;
      } else {
        cB["ij"] += (*Brandes)(A["ki"],C["kj"]);
        last_type = 0;
      }

      double t_bm = MPI_Wtime() - t_bm_st;
      if (sp_C && last_type == 0){
        nnz_out = cB.nnz_tot;
      }
      tbr.stop();
      tbrs.start();
      if (sp_C && last_type == 0){
        cB.sparsify([](cmpath p){ return p.w >= 0 && p.c != 0.0; });
      }
      tbrs.stop();
      tbra.start();
      all_cB["ij"] += (*pcB)["ij"];
      tbra.stop();
      tbrp.start();
      (*pcB)["ij"] = all_cB["ij"];
      Transform<mpath,cmpath>([](mpath p, cmpath & cp){ cp.c += 1./p.m;  })(all_B["ij"],(*pcB)["ij"]);
      if (sp_C){
        (*pcB).sparsify([](cmpath p){ return p.m == -1. && p.w != 0.; });
        if (last_type){
          cB = Matrix<cmpath>(*pcB);
          delete pcB;
        }
      } else 
        Transform<cmpath>([](cmpath & p){ if (p.m != -1 || p.w == 0.0) p = cmpath(-MAX_WHT,0,0); })(cB["ij"]);
      Transform<cmpath>([](cmpath & p){ if (p.m == -1.) p.m = 0; })(all_cB["ij"]);
      tbrp.stop();
      

      double t_all = MPI_Wtime() - t_st;
      if (!sp_B && !sp_C){
        Scalar<int> num_changed = Scalar<int>();
        num_changed[""] += Function<cmpath,int>([](cmpath p){ return p.w >= 0 && p.c!=0.0; })(cB["ij"]);
        int64_t nnz_new = num_changed.get_val();
        if (dw.rank == 0){
          printf("Brandes (dns=1) [filtered_nnz_C = %ld] <- [nnz_A = %ld] * [nnz_B = %ld] took time %lf (%lf)\n",nnz_new,A.nnz_tot,nnz_last,t_bm,t_all);
          nnz_last = nnz_new;
        }
        if (nnz_new == 0) break;
      } else {
        t_all_last = t_all;
        t_bm_last = t_bm;
      }
    }
    Transform<mpath,cmpath>([](mpath p, cmpath & cp){ if (p.w == cp.w){ cp = cmpath(p.w, p.m, cp.c*p.m); } else { cp = cmpath(p.w, p.m, 0.0); } })(all_B["ij"],all_cB["ij"]);
#ifndef TEST_SUITE
    double timbr = MPI_Wtime() - sbr;
    if (dw.rank == 0)
      printf("(%d ,%d) iter (%lf, %lf) sec\n", nbl, nbr, timbl, timbr);
#endif
    //set self-centrality scores to zero
    //FIXME: assumes loops are zero edges and there are no others zero edges in A
    Transform<cmpath>([](cmpath & p){ if (p.w == 0) p.c=0; })(all_cB["ij"]);

    //accumulate centrality scores
    v["i"] += Function<cmpath,real>([](cmpath a){ return a.c; })(all_cB["ij"]);
  }
  delete Bellman;
  delete Brandes;
}

void btwn_cnt_naive(Matrix<wht> & A, Vector<real> & v){
  World dw = *A.wrld;
  int n = A.nrow;

  Semiring<mpath> p = get_mpath_semiring();
  Monoid<cmpath> cp = get_cmpath_monoid();
  //mpath matrix to contain distance matrix
  Matrix<mpath> P(n, n, dw, p, "P");

  Function<wht,mpath> setw([](wht w){ return mpath(w, 1); });

  P["ij"] = setw(A["ij"]);
  
  Transform<mpath>([=](mpath& w){ w = mpath(MAX_WHT, 1); })(P["ii"]);

  Matrix<mpath> Pi(n, n, dw, p);
  Pi["ij"] = P["ij"];
 
  //compute all shortest mpaths by Bellman Ford 
  for (int i=0; i<n; i++){
    Transform<mpath>([=](mpath & p){ p = mpath(0,1); })(P["ii"]);
    P["ij"] = Pi["ik"]*P["kj"];
  }
  Transform<mpath>([=](mpath& p){ p = mpath(MAX_WHT, 1); })(P["ii"]);

  int lenn[3] = {n,n,n};
  Tensor<cmpath> postv(3, lenn, dw, cp, "postv");

  //set postv_ijk = shortest mpath from i to k (d_ik)
  postv["ijk"] += Function<mpath,cmpath>([](mpath p){ return cmpath(p.w, p.m, 0.0); })(P["ik"]);

  //set postv_ijk = 
  //    for all nodes j on the shortest mpath from i to k (d_ik=d_ij+d_jk)
  //      let multiplicity of shortest mpaths from i to j is a, from j to k is b, and from i to k is c
  //        then postv_ijk = a*b/c
  Transform<mpath,mpath,cmpath>(
    [=](mpath a, mpath b, cmpath & c){ 
      if (c.w<MAX_WHT && a.w+b.w == c.w){ c.c = ((real)a.m*b.m)/c.m; } 
      else { c.c = 0; }
    }
  )(P["ij"],P["jk"],postv["ijk"]);

  //sum multiplicities v_j = sum(i,k) postv_ijk
  v["j"] += Function<cmpath,real>([](cmpath p){ return p.c; })(postv["ijk"]);
}

uint64_t gen_graph(int scale, int edgef, uint64_t seed, uint64_t **edges) {

        uint64_t nedges;
        double   initiator[4] = {.57, .19, .19, .05};
        CTF::Timer tmrg("gen_graph");
        tmrg.start();
        make_graph(scale, (((int64_t)1)<<scale)*edgef, seed, seed+1, initiator, (int64_t *)&nedges, (int64_t **)edges);
        tmrg.stop();

        return nedges;
}
static void *Realloc(void *ptr, size_t sz) {

	void *lp;

	lp = (void *) realloc(ptr, sz);
	if (!lp && sz) {
		fprintf(stderr, "Cannot reallocate to %zu bytes...\n", sz);
		exit(EXIT_FAILURE);
	}
	return lp;
}

static FILE *Fopen(const char *path, const char *mode) {

	FILE *fp = NULL;
	fp = fopen(path, mode);
	if (!fp) {
		fprintf(stderr, "Cannot open file %s...\n", path);
		exit(EXIT_FAILURE);
	}
	return fp;
}


static uint64_t getFsize(FILE *fp) {

	int rv;
	uint64_t size = 0;

	rv = fseek(fp, 0, SEEK_END);
	if (rv != 0) {
		fprintf(stderr, "SEEK END FAILED\n");
		if (ferror(fp)) fprintf(stderr, "FERROR SET\n");
		exit(EXIT_FAILURE);
	}

	size = ftell(fp);
	rv = fseek(fp, 0, SEEK_SET);

	if (rv != 0) {
		fprintf(stderr, "SEEK SET FAILED\n");
		exit(EXIT_FAILURE);
	}

	return size;
}
uint64_t processedges(char **led, uint64_t ned, const int myid, uint64_t **edge) {
	int i = 0;
	uint64_t *ed=(uint64_t *)malloc(ned*sizeof(*edge));
	for (i=0; i<ned; i++) {
		uint64_t a, b;
		sscanf(led[i],"%lu %lu", &a, &b);
		ed[i*2]  = a;
		ed[i*2+1]= b;
	}
	*edge = ed;
	return ned;
}
static uint64_t read_graph_mpiio(int myid, int ntask, const char *fpath, uint64_t **edge, char ***led){
#define overlap 100
	MPI_File fh;
	MPI_Offset filesize;
	MPI_Offset localsize;
	MPI_Offset start,end;
	MPI_Status status;
	char *chunk = NULL;
	int MPI_RESULT = 0;
//	const int overlap = 100;
	uint64_t ned = 0;
	int i = 0;

	MPI_RESULT = MPI_File_open(MPI_COMM_WORLD,fpath, MPI_MODE_RDONLY, MPI_INFO_NULL,&fh);
/*	if(MPI_RESULT != MPI_SUCCESS) 
    		fprintf(stderr, exit("EXIT_FAILURE");
*/
	/* Get the size of file */
	MPI_File_get_size(fh, &filesize); //return in bytes 
	localsize = filesize/ntask;
	start = myid * localsize;
	end = start + localsize -1;
	end +=overlap;
	
	if (myid  == ntask-1) end = filesize;
	localsize = end - start + 1; //OK

	chunk = (char*)malloc( (localsize + 1)*sizeof(char)); 
	MPI_File_read_at_all(fh, start, chunk, localsize, MPI_CHAR, &status);
	chunk[localsize] = '\0';

	int locstart=0, locend=localsize;
	if (myid != 0) {
		while(chunk[locstart] != '\n') locstart++;
		locstart++;
	}
	if (myid != ntask-1) {
		locend-=overlap;
		while(chunk[locend] != '\n') locend++;
	}
	localsize = locend-locstart+1; //OK

	char *data = (char *)malloc((localsize+1)*sizeof(char));
	memcpy(data, &(chunk[locstart]), localsize);
	data[localsize] = '\0';
	free(chunk);
	
	for ( i=0; i<localsize; i++){
		if (data[i] == '\n') ned++;
	}

	(*led) = (char **)malloc(ned*sizeof(char *));
	(*led)[0] = strtok(data,"\n");

	for ( i=1; i < ned; i++)
		(*led)[i] = strtok(NULL, "\n");

	MPI_File_close(&fh);

	return ned;
}

uint64_t read_graph(int myid, int ntask, const char *fpath, uint64_t **edge) {
#define ALLOC_BLOCK     (2*1024)
#define MAX_LINE        1024

	uint64_t *ed=NULL;
	uint64_t i, j;
	uint64_t n, nmax;
	uint64_t size;
	int64_t  off1, off2;

	int64_t  rem;
	FILE     *fp;
	char     str[MAX_LINE];

	fp = Fopen(fpath, "r");

	size = getFsize(fp);
	rem = size % ntask;
	off1 = (size/ntask)* myid    + (( myid    > rem)?rem: myid);
	off2 = (size/ntask)*(myid+1) + (((myid+1) > rem)?rem:(myid+1));

	if (myid < (ntask-1)) {
		fseek(fp, off2, SEEK_SET);
		fgets(str, MAX_LINE, fp);
		off2 = ftell(fp);
	}
	fseek(fp, off1, SEEK_SET);
	if (myid > 0) {
		fgets(str, MAX_LINE, fp);
		off1 = ftell(fp);
	}

	n = 0;
	nmax = ALLOC_BLOCK; // must be even
	ed = (uint64_t *)malloc(nmax*sizeof(*ed));
	uint64_t lcounter = 0;
	uint64_t nedges = -1;
	int comment_counter = 0;

	/* read edges from file */
	while (ftell(fp) < off2) {

		// Read the whole line
		fgets(str, MAX_LINE, fp);

		// Strip # from the beginning of the line
		if (strstr(str, "#") != NULL) {
			//fprintf(stdout, "\nreading line number %"PRIu64": %s\n", lcounter, str);
			if (strstr(str, "Nodes:")) {
				//sscanf(str, "# Nodes: %" PRIu64 " Edges: %" PRIu64 "\n", &i, &nedges);
                sscanf(str, "# Nodes: %llu Edges: %llu\n", &i, &nedges);
				//fprintf(stdout, "N=%"PRIu64" E=%"PRIu64"\n", i, nedges);
			}
			comment_counter++;
		} else if (str[0] != '\0') {
			lcounter ++;
			// Read edges
//			sscanf(str, "%"PRIu64" %"PRIu64"\n", &i, &j);
	        sscanf(str, "%llu %llu\n", &i, &j);

			if (n >= nmax) {
				nmax += ALLOC_BLOCK;
				ed = (uint64_t *)Realloc(ed, nmax*sizeof(*ed));
			}
			ed[n]   = i;
			ed[n+1] = j;
			n += 2;
		}
	}
	fclose(fp);

	n /= 2; // number of ints -> number of edges
//	*edge = mirror(ed, &n); for undirected graph

	return n;
#undef ALLOC_BLOCK
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

