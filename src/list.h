#ifndef _LIST_H_
#define _LIST_H_

#include <stdlib.h>

typedef int Int_t;
typedef float Float_t;
typedef double Double_t;
typedef unsigned short bint;

struct _ListItem;

struct _ListItem {
    void* item;
    struct _ListItem* next_item;
};

typedef struct _ListItem ListItem;

ListItem* ListItem_new(void* item){
    ListItem* self = (ListItem*) malloc(sizeof(ListItem));
    self->item = item;
    self->next_item = NULL;
    return self;
}

void ListItem_delete(ListItem* self, bint delete_item){
    if (delete_item)
        free(self->item);
    free(self);
}

typedef struct {
    ListItem* first;
    ListItem* last;
    Int_t size;
} List;

inline List* List_new(void){
    List* self = (List*) malloc(sizeof(List));
    self->first = NULL;
    self->last = NULL;
    self->size = 0;
    return self;
}

inline List* List_append(List* self, void* item){
    ListItem* list_item = ListItem_new(item);
    if (self->size == 0){
        self->first = list_item;
        self->last = list_item;
        self->size = 1;
    }
    else{
        self->last->next_item = list_item;
        self->last = list_item;
        self->size += 1;
    }
}

typedef struct {
    ListItem* current;
} ListIterator;


inline void ListIterator_init(ListIterator* self, List* list){
    self->current = list->first;
}

inline void* ListIterator_next(ListIterator* self){
    ListItem* next_item = self->current;
    if (self->current != NULL)
        self->current = self->current->next_item;
    return next_item->item;
}

inline int ListIterator_has_next(ListIterator* self){
    return self->current != NULL;
}

inline void List_delete(List* self, bint delete_items){
    ListItem* current = self->first;
    ListItem* next_item;

    while (current != NULL){
        next_item = current->next_item;
        ListItem_delete(current, delete_items);
        current = next_item;
    }
    free(self);
}
#endif