package com.tyraen.voicekeyboard.feature.recognition

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.RecognitionService
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import androidx.core.content.ContextCompat
import com.tyraen.voicekeyboard.app.ServiceLocator
import com.tyraen.voicekeyboard.core.logging.DiagnosticLog
import com.tyraen.voicekeyboard.feature.audio.MicrophoneCaptureSession
import com.tyraen.voicekeyboard.feature.transcription.TranscriptionConfig
import com.tyraen.voicekeyboard.feature.transcription.WhisperPromptBuilder
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Exposes the app as an Android SpeechRecognizer provider.
 *
 * This service intentionally reuses the existing microphone capture and Whisper-compatible
 * transcription pipeline. The first version does not implement voice-activity detection; callers
 * normally stop recognition via SpeechRecognizer.stopListening(), and we also enforce a bounded
 * fallback timeout so a session cannot record forever.
 */
class DictationRecognitionService : RecognitionService() {

    companion object {
        private const val TAG = "RecognitionService"
        private const val DEFAULT_MAX_RECORDING_MS = 15_000L
        private const val MIN_RECORDING_MS = 700L
        private const val MAX_RECORDING_MS = 60_000L
        private const val RMS_NORMALIZER = 327.67f
    }

    private val scope = MainScope()
    private var activeSession: Session? = null

    private data class Session(
        val callback: Callback,
        val capture: MicrophoneCaptureSession,
        var file: File? = null,
        var finished: Boolean = false,
        var transcribing: Boolean = false,
        var timeoutJob: Job? = null,
        var processingJob: Job? = null
    )

    override fun onStartListening(recognizerIntent: Intent, listener: Callback) {
        DiagnosticLog.record(TAG, "onStartListening")

        if (activeSession != null) {
            safeError(listener, SpeechRecognizer.ERROR_RECOGNIZER_BUSY)
            return
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            safeError(listener, SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS)
            return
        }

        val session = Session(
            callback = listener,
            capture = MicrophoneCaptureSession(applicationContext)
        )
        activeSession = session

        try {
            listener.readyForSpeech(Bundle.EMPTY)
            listener.beginningOfSpeech()

            session.file = session.capture.begin { amplitude ->
                safeRmsChanged(session, normalizeRms(amplitude))
            }

            val maxRecordingMs = resolveMaxRecordingMs(recognizerIntent)
            session.timeoutJob = scope.launch {
                delay(maxRecordingMs)
                if (activeSession === session && !session.finished) {
                    DiagnosticLog.record(TAG, "Auto-stopping after ${maxRecordingMs}ms")
                    finishAndTranscribe(session)
                }
            }
        } catch (e: Exception) {
            DiagnosticLog.recordFailure(TAG, "Failed to start listening", e)
            cleanupSession(session, deleteAudio = true)
            safeError(listener, SpeechRecognizer.ERROR_AUDIO)
        }
    }

    override fun onStopListening(listener: Callback) {
        DiagnosticLog.record(TAG, "onStopListening")
        val session = activeSession
        if (session == null || session.callback !== listener) {
            safeError(listener, SpeechRecognizer.ERROR_CLIENT)
            return
        }
        finishAndTranscribe(session)
    }

    override fun onCancel(listener: Callback) {
        DiagnosticLog.record(TAG, "onCancel")
        val session = activeSession
        if (session != null && session.callback === listener) {
            cleanupSession(session, deleteAudio = true)
        }
    }

    override fun onDestroy() {
        activeSession?.let { cleanupSession(it, deleteAudio = true) }
        scope.cancel()
        super.onDestroy()
    }

    private fun finishAndTranscribe(session: Session) {
        if (session.finished || session.transcribing) return
        session.finished = true
        session.timeoutJob?.cancel()
        session.timeoutJob = null

        val callback = session.callback
        val audioFile = try {
            session.capture.finalize()
        } catch (e: Exception) {
            DiagnosticLog.recordFailure(TAG, "Failed to stop recording", e)
            cleanupSession(session, deleteAudio = true)
            safeError(callback, SpeechRecognizer.ERROR_AUDIO)
            return
        }

        try {
            callback.endOfSpeech()
        } catch (_: Exception) {
        }

        if (audioFile == null || !audioFile.exists() || audioFile.length() == 0L) {
            cleanupSession(session, deleteAudio = true)
            safeError(callback, SpeechRecognizer.ERROR_NO_MATCH)
            return
        }

        session.file = audioFile
        session.transcribing = true
        session.processingJob = scope.launch {
            try {
                val result = transcribe(audioFile, session.capture.lastDurationMs)
                if (activeSession !== session) return@launch

                result.onSuccess { text ->
                    val cleanText = text.trim()
                    if (cleanText.isBlank()) {
                        safeError(callback, SpeechRecognizer.ERROR_NO_MATCH)
                    } else {
                        callback.results(Bundle().apply {
                            putStringArrayList(
                                SpeechRecognizer.RESULTS_RECOGNITION,
                                arrayListOf(cleanText)
                            )
                            putFloatArray(
                                SpeechRecognizer.CONFIDENCE_SCORES,
                                floatArrayOf(1.0f)
                            )
                        })
                    }
                }.onFailure { error ->
                    DiagnosticLog.recordFailure(TAG, "Transcription failed", error)
                    safeError(callback, mapFailureToSpeechError(error))
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                DiagnosticLog.recordFailure(TAG, "Recognition failed", e)
                safeError(callback, mapFailureToSpeechError(e))
            } finally {
                cleanupSession(session, deleteAudio = true)
            }
        }
    }

    private suspend fun transcribe(audioFile: File, durationMs: Long): Result<String> = withContext(Dispatchers.IO) {
        val prefs = ServiceLocator.preferenceStore.load()
        if (prefs.apiKey.isBlank()) {
            return@withContext Result.failure(IllegalStateException("API key not set"))
        }

        val vocabulary = ServiceLocator.preferenceStore.loadVocabulary()
        val config = TranscriptionConfig(
            apiKey = prefs.apiKey,
            endpoint = prefs.endpoint,
            model = prefs.model,
            language = prefs.language,
            prompt = WhisperPromptBuilder.build(prefs.prompt, vocabulary),
            vocabulary = vocabulary,
            recordingDurationMs = durationMs
        )

        ServiceLocator.speechToTextClient.transcribe(audioFile, config)
            .map { text -> applyRecognitionFormatting(text, prefs.addTrailingSpace, prefs.singleWordStripPunctuation) }
    }

    private fun cleanupSession(session: Session, deleteAudio: Boolean) {
        session.timeoutJob?.cancel()
        session.processingJob?.cancel()
        session.timeoutJob = null
        session.processingJob = null

        try {
            session.capture.release()
        } catch (_: Exception) {
        }

        if (deleteAudio) {
            try {
                session.file?.delete()
            } catch (_: Exception) {
            }
        }

        if (activeSession === session) {
            activeSession = null
        }
    }

    private fun applyRecognitionFormatting(
        text: String,
        addTrailingSpace: Boolean,
        singleWordStripPunctuation: Boolean
    ): String {
        var output = if (singleWordStripPunctuation) stripSingleWordPunctuation(text) else text
        if (addTrailingSpace) output += " "
        return output
    }

    private fun stripSingleWordPunctuation(text: String): String {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) return text
        if (trimmed.any { it.isWhitespace() }) return text
        return trimmed.trimEnd('.', '!', '?', '。', '！', '？')
    }

    private fun resolveMaxRecordingMs(intent: Intent): Long {
        val requested = intent.getLongExtra(
            RecognizerIntent.EXTRA_SPEECH_INPUT_MINIMUM_LENGTH_MILLIS,
            DEFAULT_MAX_RECORDING_MS
        )
        return requested.coerceIn(MIN_RECORDING_MS, MAX_RECORDING_MS)
    }

    private fun normalizeRms(amplitude: Int): Float =
        (amplitude / RMS_NORMALIZER).coerceAtMost(100.0f)

    private fun safeRmsChanged(session: Session, value: Float) {
        try {
            if (activeSession === session && !session.finished) {
                session.callback.rmsChanged(value)
            }
        } catch (_: Exception) {
        }
    }

    private fun safeError(callback: Callback, code: Int) {
        try {
            callback.error(code)
        } catch (_: Exception) {
        }
    }

    private fun mapFailureToSpeechError(error: Throwable): Int {
        val message = error.message.orEmpty()
        return when {
            message.contains("API key", ignoreCase = true) -> SpeechRecognizer.ERROR_CLIENT
            message.contains("401") || message.contains("403") -> SpeechRecognizer.ERROR_CLIENT
            message.contains("timeout", ignoreCase = true) -> SpeechRecognizer.ERROR_NETWORK_TIMEOUT
            message.contains("network", ignoreCase = true) || message.contains("server", ignoreCase = true) -> SpeechRecognizer.ERROR_NETWORK
            else -> SpeechRecognizer.ERROR_SERVER
        }
    }
}
