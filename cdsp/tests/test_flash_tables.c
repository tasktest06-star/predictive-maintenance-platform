/* Does the flash-table pipeline produce the same features as the RAM one?
 *
 * PDM_TABLES_IN_FLASH swaps runtime-computed window/twiddle/bit-reversal tables
 * for precomputed `const` ones, saving 46 kB of RAM on a 4096-point pipeline --
 * which on the 256 kB nRF9160 is the difference between fitting comfortably and
 * not fitting. The precomputation is done by cdsp/tools/gen_tables.py using the
 * same formulas the runtime uses.
 *
 * "Same formulas" is not the same as "same bits". If precomputation introduced
 * any drift, every threshold tuned on one build would be subtly wrong on the
 * other, and the two builds are deployed to different kits on the same fleet.
 * This test compares the flash-table build's feature vector against the golden
 * vector the RAM build is already checked against.
 *
 * Build and run: make -C cdsp test-flash-tables
 */
#include "pdm_pipeline.h"
#include "golden.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
int main(void){
    if (pdm_pipeline_init()!=PDM_OK){printf("init failed\n");return 1;}
    pdm_pipeline_result r; memset(&r,0,sizeof(r)); r.threshold_hint=0.0f;
    if (pdm_pipeline_run(gold_kit_accel, GOLD_KIT_NACQ, GOLD_KIT_SHAFT, &r)!=PDM_OK){
        printf("run failed\n"); return 1; }
    int bad=0; double worst=0;
    for (int i=0;i<GOLD_KIT_BEARING_IDX;++i){
        double g=r.features[i], want=gold_kit_features[i];
        double rel=fabs(g-want)/(fabs(want)>1e-9?fabs(want):1.0);
        if (rel>worst) worst=rel;
        if (rel>5e-3) {++bad; if(bad<4) printf("  feature %d: %.6e vs %.6e (rel %.2e)\n",i,g,want,rel);}
    }
    printf("flash tables: tier %u, %d of %d features outside 5e-3, worst %.2e -> %s\n",
           (unsigned)r.tier, bad, GOLD_KIT_BEARING_IDX, worst, bad? "DIFFERS":"IDENTICAL");
    return bad?1:0;
}
