package com.example.sanraksha

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.launch

data class ApiPredictionResult(
    val prediction: Int,
    val label: String
)

class HealthViewModel : ViewModel() {

    fun sendHealthDataToApi(inputs : healthDataItem,onResult : (ApiPredictionResult?)->Unit){
        val healthDataList = listOf(
            inputs
        )
        viewModelScope.launch {
            try{
                val response = retrofitInstance.api.sendHealthData(healthDataList)
                if (response.isSuccessful) {
                    val body = response.body()
                    val prediction = body?.prediction?.firstOrNull()
                    if (prediction != null) {
                        val label = body.prediction_label?.firstOrNull()
                            ?: when (prediction) {
                                0 -> "Low Risk"
                                1 -> "High Risk"
                                else -> "Unknown"
                            }
                        Log.d("API_SUCCESS", "Prediction: $prediction, Label: $label")
                        onResult(ApiPredictionResult(prediction = prediction, label = label))
                    } else {
                        onResult(null)
                    }
                } else {
                    Log.e("API_ERROR", "Response Code: ${response.code()}")
                    Log.e("API_ERROR", "Error Body: ${response.errorBody()?.string()}")
                    onResult(null)
                }

            }catch(e : Exception){
                Log.d("API_REQUEST", "Sending data: $healthDataList")
                Log.e("API_EXCEPTION", "Exception: ${e.message}", e)
               onResult(null)
        }
        }

    }

}