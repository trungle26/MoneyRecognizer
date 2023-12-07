package com.kimminh.moneysense.ui.home

import android.util.Log
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mlkit.common.model.LocalModel
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.label.ImageLabeling
import com.google.mlkit.vision.label.custom.CustomImageLabelerOptions
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

@OptIn(ExperimentalGetImage::class)
class MoneyAnalyzer(private val listener: LabelListener) : ImageAnalysis.Analyzer {
    private val localModel = LocalModel.Builder()
        .setAssetFilePath("model.tflite")
        .build()
    private val customImageLabelerOptions = CustomImageLabelerOptions.Builder(localModel)
        .setConfidenceThreshold(0.95f)
        .setMaxResultCount(5)
        .build()
    private val labeler = ImageLabeling.getClient(customImageLabelerOptions)
    val labelList = mutableListOf<String>("0","1.000VND","5.000VND","10.000VND","20.000VND","50.000VND")

    override fun analyze(imageProxy: ImageProxy) {
        val mediaImage = imageProxy.image
        if (mediaImage != null) {
            val image = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)
            // Pass image to an ML Kit Vision API
            labeler.process(image)
                .addOnSuccessListener { labels ->
                    for (label in labels) {
                        val text = labelList[label.index]
                        val confidence = label.confidence
                        Log.d("Confidence", "$text: $confidence")
                        listener(text.substringAfter(' '))
                    }
                }
                .addOnFailureListener { e ->
                    Log.e("Classification Error", e.toString())
                }
                .addOnCompleteListener {
                    CoroutineScope(Dispatchers.IO).launch {
                        delay(2000)
                        imageProxy.close()
                    }
                }
        }
    }
}