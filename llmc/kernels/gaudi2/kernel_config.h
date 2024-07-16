#if defined(FLOAT32)
#define M_PI 3.14f
#define GELU_SCALING_FACTOR v_rsqrt_f32(2.0f / M_PI)
#define VECTOR                      float
#define VECTOR_SIZE                 64
typedef float                       SCALAR;
#define floatX                      float; 
#define v_ld_tnsr_i(a,b)            v_f32_ld_tnsr_b(a,b)
#define v_ld_tnsr_partial_i(a,b, c, d)    v_f32_ld_tnsr_partial_b(a, b, c, d)
#define v_reduce_max(a)             v_f32_reduce_max(a)
#define v_reduce_sum(a)         v_f32_reduce_sum(a)
#define v_sel_less_v_s_v_v(a,b,c,d) v_f32_sel_less_f32_b(a,b,c,d)
#define v_sel_geq_v_s_v_v(a,b,c,d)    v_f32_sel_geq_f32_b(a,b,c,d)
#define v_sel_grt_v_s_v_v(a,b,c,d)    v_f32_sel_grt_f32_b(a,b,c,d)
#define v_sel_leq_v_s_v_v(a, b, c, d) v_f32_sel_leq_f32_b(a, b, c, d)
#define v_sel_geq_v_s_v_v_b(a, b, c, d, i, p, o)  v_f32_sel_geq_f32_b(a, b, c, d, 0, i, p, o)
#define v_sel_less_v_s_v_v_b(a, b, c, d, i, p, o) v_f32_sel_less_f32_b(a, b, c, d, 0, i, p, o)
#define st_tnsr_i_v(a,b,c)          v_f32_st_tnsr(a,b,c)
#define bv_cmp_eq_v_v(a, b)         from_bool64(v_f32_cmp_eq_b(a, b))
#define bv_u_cmp_geq_v_s(a, b)      from_bool64(v_u32_cmp_geq_b(a, b))
#define v_mov_v_vb(a, b, c, d)      v_f32_mov_vb(a, 0, b, to_bool64(c), d)
#define v_mov_s_vb(a, b, c, d)      v_f32_mov_vb(a, 0, b, to_bool64(c), d)
#define v_add_v_v_b(a, b, c, d, e)  v_f32_add_b(a, b, 0, c, d, e)
#define v_mul_v_s(a, b)             v_f32_mul_b(a, b)
#define v_mul_v_v(a,b)              v_f32_mul_b(a, b)
#define s_ld_g_a(a)                 s_f32_ld_g(a)
#define v_ld_g_a(a)                 v_f32_ld_g(a)
#define v_mov_s(a)                  v_f32_mov_b(a)
#define log(a)                      log_f32(a)
#define exp(a)                      exp_f32(a)
#define min(a, b)                   s_f32_min(a, b)
#define V_LANE_ID                   read_lane_id_4b_b()
#define sqrt(a)                     v_rsqrt_f32(a)
#define add(a, b)                   s_f32_add(a, b)
#define v_sel_leq_v_v_v_v(a, b, c, d) v_f32_sel_leq_f32_b(a, b, c, d)

#endif

#if defined(BFLOAT16)
#define VECTOR                      bfloat128
#define VECTOR_SIZE                 128
typedef bf16                        SCALAR;
#define v_ld_tnsr_i(a,b)            v_bf16_ld_tnsr_b(a,b)
#define v_sel_less_v_s_v_v(a,b,c,d) v_bf16_sel_less_bf16_b(a,b,c,d)
#define v_sel_geq_v_s_v_v(a,b,c,d)    v_bf16_sel_geq_bf16_b(a,b,c,d)
#define v_sel_grt_v_s_v_v(a,b,c,d)    v_bf16_sel_grt_bf16_b(a,b,c,d)
#define v_sel_leq_v_s_v_v(a, b, c, d) v_bf16_sel_leq_bf16_b(a, b, c, d)
#define v_sel_geq_v_s_v_v_b(a, b, c, d, i, p, o)  v_bf16_sel_geq_bf16_b(a, b, c, d, 0, i, p, o)
#define v_sel_less_v_s_v_v_b(a, b, c, d, i, p, o) v_bf16_sel_less_bf16_b(a, b, c, d, 0, i, p, o)
#define st_tnsr_i_v(a,b,c)          v_bf16_st_tnsr(a,b,c)

#endif