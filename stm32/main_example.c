/*
 * main_example.c - how to use drone_model.h on an STM32.
 *
 * This is NOT a complete project; it is a minimal usage sketch. You provide
 * the raw 16 kHz audio samples (e.g. from an I2S/PDM microphone via DMA),
 * then call drone_detect() to get the probability that a drone is present.
 *
 * The library is STREAMING: extract_feature_vector() keeps only ~5 KB of its
 * own RAM (a per-frame FFT scratch + running accumulators). You still need to
 * hold the audio in YOUR buffer, which the function reads twice (once to find
 * the peak, once to compute features).
 *
 * Memory math (STM32C092: 30 KB SRAM):
 *   - float32 audio costs 64 KB/second, so a full 2 s clip (128 KB) does NOT fit.
 *   - For a demo, feed a shorter window (0.25 s = 16 KB) or store int16.
 *   - For continuous real-time detection, replace the global peak normalisation
 *     with a fixed gain / AGC so you never need the whole clip at once.
 */
#include "drone_model.h"

#if defined(STM32)
#include "stm32f4xx_hal.h"
#endif

/* Your microphone driver fills this with float samples in [-1.0, 1.0] at
 * 16 kHz. 0.25 s = 4000 samples = 16 KB (fits alongside the library's ~5 KB
 * inside 30 KB SRAM). Increase only if your part has more RAM. */
static float g_audio[SAMPLE_RATE / 4];
static int   g_audio_len = 0;

/* Call this once you have >= FRAME_SAMPLES samples buffered. */
void drone_detect_run(void)
{
    if (g_audio_len < FRAME_SAMPLES) {
        return;  /* not enough audio yet */
    }

    float prob = drone_detect(g_audio, g_audio_len);

#if defined(STM32)
    /* Simple LED output: on = drone, off = no drone. */
    if (prob > 0.5f) {
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_SET);
    } else {
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_RESET);
    }
#endif

    /* For printf-based debugging (needs a debug UART). */
    /* printf("drone probability = %.2f\r\n", (double)prob); */
}
