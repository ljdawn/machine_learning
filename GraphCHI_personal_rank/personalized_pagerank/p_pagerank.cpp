#include <string>
#include <fstream>
#include <cmath>

#define GRAPHCHI_DISABLE_COMPRESSION


#include "graphchi_basic_includes.hpp"
#include "util/toplist.hpp"

using namespace graphchi;

#define THRESHOLD 1e-1    
#define RANDOMRESETPROB 0.15


typedef float VertexDataType;
typedef float EdgeDataType;

unsigned int root;
int scale;
float dang_sum;
float total;
struct PagerankProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {
    void before_iteration(int iteration, graphchi_context &info) {}
    void after_iteration(int iteration ,graphchi_context &ginfo) {}
    void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &ginfo) {}

    void update(graphchi_vertex<VertexDataType, EdgeDataType> &v, graphchi_context &ginfo) {
        float sum=0;
        if (ginfo.iteration == 0) {
                for(int i=0; i < v.num_outedges(); i++) {
                        graphchi_edge<float> * edge = v.outedge(i);
                        edge->set_data(1.0/v.num_outedges());
                        std::cout<<"node:"<<v.id()<<" val:"<<v.get_data()<<" n_edge:"<<i<<" "<<v.num_outedges()<<" val:"<<edge->get_data()<<std::endl;
                        }
                scale += 1;
        }else if (ginfo.iteration == 1) {
                v.set_data(1.0/scale);
                std::cout<<"node:"<<v.id()<<" pr_val:"<<v.get_data()<<std::endl;
        }else if (ginfo.iteration %4 == 2) {
                total = 0.0;
                float dang = 0.0;
                if (v.num_outedges() == 0){
                        dang = v.get_data();
                        dang_sum += ((1-RANDOMRESETPROB)*dang*(1.0/scale));
                        std::cout <<"scale: "<<scale<<"dang: "<<dang<<std::endl;
                }
                std::cout<<"dang val:"<<dang<<" dang_sum val:"<<dang_sum<<std::endl;
        }else if(ginfo.iteration %4 == 3){
            int sw = 0;
            if (v.id() == root) {sw = 1;}

            for(int i=0; i < v.num_inedges(); i++) {
                float val = v.inedge(i)->get_data();
                sum += val;
            }
            float pagerank = RANDOMRESETPROB*sw + sum + dang_sum;

            v.set_data(pagerank);
            std::cout<<"node id: "<<v.id()<<" pr_val: "<<v.get_data()<<std::endl;

            for(int i=0; i < v.num_outedges(); i++) {
                graphchi_edge<float> * edge = v.outedge(i);
                edge->set_data((v.get_data() * (1 - RANDOMRESETPROB))/v.num_outedges());
            }
        }else if(ginfo.iteration %4 == 0){
                dang_sum = 0.0;
                total += v.get_data();
        }else if(ginfo.iteration %4 == 1){
                v.set_data(v.get_data()/total);
                std::cout<<"node_f val:"<<v.id()<<" pagerank val:"<<v.get_data()<<std::endl;
        }
    }

};

int main(int argc, const char ** argv) {
    graphchi_init(argc, argv);
    metrics m("pagerank");

    /* Parameters */
    std::string filename    = get_option_string("file"); // Base filename
    int niters              = get_option_int("niters", 10);
    bool scheduler          = false;                    // Non-dynamic version of pagerank.
    int ntop                = get_option_int("top", 20);
    root                    = get_option_int("root", 0);

    /* Process input file - if not already preprocessed */
    //int nshards             = convert_if_notexists<EdgeDataType>(filename, get_option_string("nshards", "auto"));
    int nshards             = convert_if_notexists<EdgeDataType>(filename, get_option_string("nshards", "1"));

    /* Run */
    graphchi_engine<float, float> engine(filename, nshards, scheduler, m);
    engine.set_modifies_inedges(false); // Improves I/O performance.
    PagerankProgram program;
    engine.run(program, niters);

    /* Output top ranked vertices */
    std::vector< vertex_value<float> > top = get_top_vertices<float>(filename, ntop);
    std::cout << "Print top " << ntop << " vertices:" << std::endl;
    for(int i=0; i < (int)top.size(); i++) {
        std::cout << (i+1) << ". " << top[i].vertex << "\t" << top[i].value << std::endl;
    }

//    metrics_report(m);    
    return 0;
}
                                                  