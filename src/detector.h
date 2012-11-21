#include "list.h"

typedef struct {
    Int_t x1, y1, x2, y2;
    Float_t weight;
} Rect;

const Int_t TREE_LEFT = 0;
const Int_t TREE_RIGHT = 1;

inline void Rect_init(Rect* self, Int_t x1, Int_t y1, Int_t x2, Int_t y2, Float_t weight){
    self->x1 = x1;
    self->y1 = y1;
    self->x2 = x2;
    self->y2 = y2;
    self->weight = weight;
}


inline Rect* Rect_new(Int_t x1, Int_t y1, Int_t x2, Int_t y2, Float_t weight){
    Rect* self = (Rect*) malloc(sizeof(Rect));
    Rect_init(self, x1, y1, x2, y2, weight);
    return self;
}

inline void Rect_delete(Rect* self){
    free(self);
}

typedef struct {
    Rect** rects;
    Int_t nb_rects;
    Float_t threshold;
    Float_t left_val;
    Float_t right_val;
    Float_t width;
    Float_t height;
    Int_t left_node;
    Int_t right_node;
    bint has_left_val;
    bint has_right_val;
} Feature;

void inline Feature_init(Feature* self,
                  List* rect_list,
                  Float_t threshold,
                 Float_t left_val , Int_t left_node , bint has_left_val,
                 Float_t right_val, Int_t right_node, bint has_right_val,
                 Float_t width, Float_t height){
    Int_t i;
    ListIterator it;

    self->nb_rects = rect_list->size;
    self->rects = (Rect**) calloc(self->nb_rects, sizeof(Rect*));

    ListIterator_init(&it, rect_list);
    for (i=0; i< self->nb_rects; i++)
        self->rects[i] = (Rect*)ListIterator_next(&it);
    
    self->threshold = threshold;
    self->left_val = left_val;
    self->right_val = right_val;
    self->width = width;
    self->height = height;
    self->left_node = left_node;
    self->right_node = right_node;
    self->has_left_val = has_left_val;
    self->has_right_val = has_right_val;
}

inline Feature* Feature_new(List* Rectlist,
                 Float_t threshold,
                 Float_t left_val , Int_t left_node , bint has_left_val,
                 Float_t right_val, Int_t right_node, bint has_right_val,
                 Float_t width, Float_t height){
    Feature* self = (Feature*) malloc(sizeof(Feature));
    Feature_init(self, Rectlist,
                 threshold,
                 left_val , left_node , has_left_val,
                 right_val, right_node, has_right_val,
                 width, height);
    return self;
}
    
    
    
inline void Feature_delete(Feature* self){
    Int_t i;
    
    if (self->rects != NULL){
        for (i=0; i<self->nb_rects; i++)
            Rect_delete(self->rects[i]);
        free(self->rects);
    }
    free(self);
}

inline Int_t Feature_getLeftOrRight(Feature* self,
                             Int_t width,
                             Int_t height,
                             Int_t* gray_image,
                             Int_t* squares,
                             Int_t i, Int_t j, Float_t scale){
    

        Int_t w = (Int_t) (scale * self->width);
        Int_t h = (Int_t) (scale * self->height);
        Double_t inv_area = 1.0/(w*h);
        
        Int_t total_x  = gray_image[(i+h)*width + (j+w)] + gray_image[i * width + j] 
                    - gray_image[i * width + (j+w)] - gray_image[(i+h) * width + j];

        Int_t total_x2 = squares[(i+h) * width + (j+w)] + squares[i * width + j]
                    - squares[i * width +  (j+w)] - squares[(i+h) * width + j];

        Double_t moy = total_x * inv_area;
        Double_t vnorm = total_x2 * inv_area-moy * moy;
        Int_t k, rx1, rx2, ry1, ry2;
        Rect* r;
        Int_t Rectsum;
        Double_t Rectsum2;
        Int_t resp;
        
        vnorm = (vnorm > 1 ? sqrt(vnorm) : 1.0);
        
        Rectsum = 0;

        for (k=0; k<self->nb_rects; k++){
            r = self->rects[k];
            rx1 = j + (Int_t) (scale*r->x1);
            rx2 = j + (Int_t) (scale*(r->x1+r->y1));
            ry1 = i + (Int_t) (scale*r->x2);
            ry2 = i + (Int_t) (scale*(r->x2+r->y2));
            Rectsum += (Int_t)((gray_image[ry2 * width + rx2]
                                -gray_image[ry1 * width + rx2]
                                -gray_image[ry2 * width + rx1]
                                +gray_image[ry1 * width + rx1])*r->weight);
        }
        Rectsum2 = Rectsum * inv_area;

        if (Rectsum2 < self->threshold * vnorm)
            resp = TREE_LEFT;
        else
            resp = TREE_RIGHT;
        return resp;
}

typedef struct {
    Feature**  features;
    Int_t nfeatures;
} Tree;

inline void Tree_init(Tree* self, List* feature_list){
    
    Int_t i;
    ListIterator it;
        
    self->nfeatures = feature_list->size;
    self->features = (Feature **) calloc(self->nfeatures, sizeof(Feature*));
    
    ListIterator_init(&it, feature_list);
    for (i=0; i < self->nfeatures; i++)
        self->features[i] = (Feature*) ListIterator_next(&it);
}

inline Tree* Tree_new(List* feature_list){
    Tree* self = (Tree*) malloc(sizeof(Tree));
    Tree_init(self, feature_list);
    return self;
}
    
inline void Tree_delete(Tree* self){
    Int_t i;
    if (self->features != NULL){
        for (i=0; i<self->nfeatures; i++)
            Feature_delete(self->features[i]);
        free(self->features);
    }
    free(self);
}
  
inline Float_t Tree_get_val(Tree* self,
        Int_t width, Int_t height,
        Int_t* gray_image,
        Int_t* squares,
        int i, int j, Float_t scale){

    Feature* cur_node = self->features[0];
    Int_t where;
        
    while (1){
        where = Feature_getLeftOrRight(cur_node, width, height, gray_image, squares, i, j, scale);
        if (where == TREE_LEFT){
            if (cur_node->has_left_val)
                return cur_node->left_val;
            else
                cur_node = self->features[cur_node->left_node];
        }
        else{
            if (cur_node->has_right_val)
                return cur_node->right_val;
            else
                cur_node = self->features[cur_node->right_node];
        }
    }
}

typedef struct {
    Tree** trees;
    Int_t nb_trees;
    Float_t threshold;
} Stage;

inline void Stage_init(Stage* self, List* tree_list, Float_t threshold){
    
    Int_t i;
    ListIterator it;
    
    self->nb_trees = tree_list->size;
    self->trees = (Tree**) calloc(self->nb_trees, sizeof(Tree*));
    ListIterator_init(&it, tree_list);
    for (i=0; i<self->nb_trees; i++)
        self->trees[i] = (Tree*) ListIterator_next(&it);
    
    self->threshold = threshold;
}

inline Stage* Stage_new(List* tree_list, Float_t threshold){
    Stage* self = (Stage*) malloc(sizeof(Stage));
    Stage_init(self, tree_list, threshold);
    return self;
}

inline void Stage_delete(Stage* self){
    Int_t i;
    
    if (self->trees != NULL){
        for (i=0; i<self->nb_trees; i++){
            Tree_delete(self->trees[i]);
        }
        free(self->trees);
    }
    free(self);
}


inline bint Stage_pass(Stage* self, 
        Int_t width, Int_t height,
        Int_t* gray_image,
        Int_t* squares,
        Int_t i, Int_t j, Float_t scale){
    Float_t sum_ = 0;
    Int_t k;

    for (k=0; k<self->nb_trees;k++)
        sum_ += Tree_get_val(self->trees[k],
                width, height,
                gray_image, squares, i, j, scale);

    return sum_ > self->threshold;
}


void get_integral_canny(
    const Int_t width,
    const Int_t height,
    Int_t* gray_image,
    Int_t* canny){
        
    Int_t grad[height * width ];
    Int_t sum_;
    Int_t i,j;
    Int_t grad_x, grad_y, col, value;

        for (i=2; i<height-2; i++){
            for (j=2; j<width-2; j++){
                sum_ = 0;
                sum_ += 2  * gray_image[(i-2)* width + j-2];
                sum_ += 4  * gray_image[(i-2)* width + j-1];
                sum_ += 5  * gray_image[(i-2)* width + j+0];
                sum_ += 4  * gray_image[(i-2)* width + j+1];
                sum_ += 2  * gray_image[(i-2)* width + j+2];
                sum_ += 4  * gray_image[(i-1)* width + j-2];
                sum_ += 9  * gray_image[(i-1)* width + j-1];
                sum_ += 12 * gray_image[(i-1)* width + j+0];
                sum_ += 9  * gray_image[(i-1)* width + j+1];
                sum_ += 4  * gray_image[(i-1)* width + j+2];
                sum_ += 5  * gray_image[(i+0)* width + j-2];
                sum_ += 12 * gray_image[(i+0)* width + j-1];
                sum_ += 15 * gray_image[(i+0)* width + j+0];
                sum_ += 12 * gray_image[(i+0)* width + j+1];
                sum_ += 5  * gray_image[(i+0)* width + j+2];
                sum_ += 4  * gray_image[(i+1)* width + j-2];
                sum_ += 9  * gray_image[(i+1)* width + j-1];
                sum_ += 12 * gray_image[(i+1)* width + j+0]  ;  
                sum_ += 9  * gray_image[(i+1)* width + j+1];
                sum_ += 4  * gray_image[(i+1)* width + j+2];
                sum_ += 2  * gray_image[(i+2)* width + j-2];
                sum_ += 4  * gray_image[(i+2)* width + j-1];
                sum_ += 5  * gray_image[(i+2)* width + j+0];
                sum_ += 4  * gray_image[(i+2)* width + j+1];
                sum_ += 2  * gray_image[(i+2)* width + j+2];
                
                canny[i * width + j] = sum_/159;
            }
        }

        for (i=1; i<height-1; i++){
            for (j=1; j<width-1; j++){
                grad_x = - canny[(i-1) * width + j-1] + canny[(i+1) * width + j-1]
                         - 2 * canny[(i-1) * width  + j] + 2 * canny[(i+1) * width + j]
                         - canny[(i-1) * width + j+1] + canny[(i+1) * width + j+1];
                
                grad_y = canny[(i-1)* width + j-1] + 2 * canny[i* width + j-1] 
                        + canny[(i+1)* width + j-1] - canny[(i-1)* width + j+1] 
                        - 2 * canny[i* width + j+1] - canny[(i+1)* width + j+1];

                grad[i* width + j] = (Int_t) (fabs(grad_x) + fabs(grad_y));
            }
        }

        for (i=0; i<height; i++){
            col = 0;
            for (j=0; j<width; j++){
                value = grad[i* width + j];
                canny[i* width + j] = (i > 0 ? canny[(i-1)* width + j] : 0) + col + value;
                col += value;
            }
        }
}

List* det_get_objects(
            List* stages,
            Float_t dwidth,
            Float_t dheight,
            Int_t width,
            Int_t height,
            Int_t* img, 
            Float_t scale_base, Float_t scale_inc,
            Float_t increment, bint do_canny_pruning){
    
        List* ret = List_new();
        Int_t *ret_val;
        ListIterator it;
        Float_t max_scale = fmin(((float)width )/dwidth,
                                 ((float)height)/dheight );
        
        Int_t gray_image[height * width];
        Int_t squares[height * width];
        Int_t* canny;

        Int_t i, j;
        Int_t col, col2, value;
        Int_t step, size;
        Int_t edges_density, d;
        bint do_pass;
        Float_t scale;

        for (i=0; i<height; i++){
            col = 0;
            col2 = 0;
            for (j=0; j<width; j++){
                value = img[i * width + j];
                gray_image[i * width + j] = 
                    (i>0 ? gray_image[(i-1) * width + j] : 0) + col + value;
                squares[i* width + j] =
                    (i>0 ? squares[(i-1) * width + j] : 0) + col2 + value * value;
                col += value;
                col2 += value * value;
            }
        }

        if (do_canny_pruning){
            canny = (Int_t*)calloc(height * width, sizeof(Int_t));
            get_integral_canny(width, height, img, canny);
        }
        
        scale = scale_base;
        while (scale < max_scale){
            step = (Int_t) (scale * dwidth * increment);
            size = (Int_t) (scale * dheight);
            
            for (i=0; i<height - size; i +=step){
                for (j=0; j<width - size; j += step){
                    if (do_canny_pruning){
                        edges_density = canny[(i+size) * width + j+size] + canny[i*width + j] -
                                        canny[i*width +  j+size]-canny[(i+size) * width + j];
                        d = edges_density/size/size;
                        if (d < 20 || d > 100)
                            continue;
                    }
                        
                    do_pass = 1;
                    
                    ListIterator_init(&it, stages);
                    
                    while (ListIterator_has_next(&it)){
                        if (! Stage_pass((Stage*)ListIterator_next(&it),
                                          width,
                                          height,
                                          gray_image, 
                                          squares, i, j, scale) ){
                            do_pass = 0;
                            break;
                        }
                    }

                    if (do_pass){
                        ret_val = (Int_t*) calloc(4, sizeof(Int_t));
                        ret_val[0] = j;
                        ret_val[1] = i;
                        ret_val[2] = size;
                        ret_val[3] = size;
                        List_append(ret, ret_val);
                    }
                }
            }
            
            scale *= scale_inc;
        }
        if (do_canny_pruning) free(canny);
    return ret;
}
