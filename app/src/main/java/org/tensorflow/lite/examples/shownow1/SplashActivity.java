package org.tensorflow.lite.examples.shownow1;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.speech.tts.TextToSpeech;
import android.util.Log;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.examples.shownow1.R;

import java.util.Locale;

public class SplashActivity extends AppCompatActivity {

    private TextToSpeech mTTS;

    @Override
    protected void onCreate(Bundle savedInstanceStare) {
        super.onCreate(savedInstanceStare);
        setContentView(R.layout.activity_splash);

        mTTS = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if (status == TextToSpeech.SUCCESS) {
                    int result = mTTS.setLanguage(Locale.KOREAN);
                    if (result == TextToSpeech.LANG_MISSING_DATA ||
                            result == TextToSpeech.LANG_NOT_SUPPORTED) {
                        Log.e("TTS", "Language not supported");
                    }
                } else {
                    Log.e("TTS", "Initialization failed");
                }
            }
        });


        Handler handler = new Handler ();
        handler.postDelayed(new Runnable() {

            private void speak(String text) {
                mTTS.speak(text, TextToSpeech.QUEUE_FLUSH, null);
            }
            @Override
            public void run() {
                Intent intent = new Intent(getApplicationContext(),HomeActivity.class);
                startActivity(intent);
                mTTS.setSpeechRate(4.0f);
                speak("안녕하세요쇼우나우 입니다, 설명을 들으시려면 화면중앙의 설명듣기 버튼을, 시작하시려면 하단의 시작하기 버튼을 눌러주세요");
                finish();
            }
        },2000); // 2초 있다 메인액티비티로
    }

    @Override
    protected void onPause() {
        super.onPause();
        finish();
    }
}

