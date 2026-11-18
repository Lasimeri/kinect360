/* tribuf.h: single-producer single-consumer triple buffer in one atomic word.
 * state = index of the middle buffer | 4 when it holds a frame the consumer has not taken.
 * The producer owns `back`, the consumer owns `front`; ownership moves only through the exchange. */
#pragma once
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct { uint8_t *buf[3]; _Atomic unsigned state; unsigned back, front; } tribuf;

static inline int tb_init(tribuf *t, size_t bytes)
{
	for (int i = 0; i < 3; i++)
		if (!(t->buf[i] = calloc(1, bytes))) return -1;
	t->back = 0;
	t->front = 2;
	atomic_init(&t->state, 1u);
	return 0;
}

/* producer: back is complete; it becomes the fresh middle, the old middle becomes back */
static inline uint8_t *tb_publish(tribuf *t)
{
	unsigned old = atomic_exchange_explicit(&t->state, t->back | 4u, memory_order_acq_rel);
	t->back = old & 3u;
	return t->buf[t->back];
}

/* consumer: slot index of a fresh frame (now front), or -1 when nothing is new */
static inline int tb_acquire_slot(tribuf *t)
{
	if (!(atomic_load_explicit(&t->state, memory_order_acquire) & 4u)) return -1;
	unsigned old = atomic_exchange_explicit(&t->state, t->front, memory_order_acq_rel);
	t->front = old & 3u;
	return (int)t->front;
}

static inline const uint8_t *tb_acquire(tribuf *t)
{
	int s = tb_acquire_slot(t);
	return s < 0 ? NULL : t->buf[s];
}
