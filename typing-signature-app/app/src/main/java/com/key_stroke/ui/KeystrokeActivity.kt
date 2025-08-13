package com.hyunsung.key_stroke.ui

import android.annotation.SuppressLint
import android.os.Bundle
import android.util.Log
import android.webkit.JavascriptInterface
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Toast
import androidx.activity.ComponentActivity
import com.hyunsung.key_stroke.data.Constants

class KeystrokeActivity : ComponentActivity() {

    companion object {
        private const val TAG = "KeystrokeActivity"
    }

    private var loginId: String = ""

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        loginId = intent.getStringExtra(Constants.EXTRA_LOGIN_ID) ?: ""
        Log.d(TAG, "KeystrokeActivity created, loginId=$loginId")

        val webView = WebView(this).apply {
            settings.apply {
                javaScriptEnabled = true
                domStorageEnabled = true
                allowFileAccess = true
                allowContentAccess = true
                mixedContentMode = android.webkit.WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
                cacheMode = android.webkit.WebSettings.LOAD_NO_CACHE
                loadsImagesAutomatically = true
                blockNetworkImage = false
                blockNetworkLoads = false
                allowUniversalAccessFromFileURLs = true
                allowFileAccessFromFileURLs = true
            }

            addJavascriptInterface(object {
                @JavascriptInterface
                fun getLoginId(): String {
                    return loginId
                }

                @JavascriptInterface
                fun startKeystrokeAuth() {
                    Log.d(TAG, "startKeystrokeAuth called")
                    runOnUiThread {
                        loadUrl(Constants.KEYSTROKE_AUTH_HTML_PATH)
                    }
                }

                @JavascriptInterface
                fun navigateToMainPage() {
                    Log.d(TAG, "navigateToMainPage called")
                    runOnUiThread {
                        loadUrl("file:///android_asset/main_page.html")
                    }
                }

                @JavascriptInterface
                fun showToast(message: String) {
                    runOnUiThread {
                        Toast.makeText(this@KeystrokeActivity, message, Toast.LENGTH_SHORT).show()
                    }
                }

                @JavascriptInterface
                fun logError(message: String) {
                    Log.e(TAG, message)
                }
            }, Constants.JAVASCRIPT_INTERFACE_NAME)

            webViewClient = object : WebViewClient() {
                override fun onPageFinished(view: WebView?, url: String?) {
                    super.onPageFinished(view, url)
                    // 주입: 로그인 아이디 전달
                    val idEscaped = loginId.replace("'", "\\'")
                    view?.evaluateJavascript("if (window.setLoginId) { window.setLoginId('$idEscaped'); } else { window.__LOGIN_ID='$idEscaped'; }", null)
                }
            }

            loadUrl(Constants.AUTH_SELECTION_HTML_PATH)
        }

        setContentView(webView)
    }
}


