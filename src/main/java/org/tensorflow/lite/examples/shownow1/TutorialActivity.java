package org.tensorflow.lite.examples.shownow1;

import static android.speech.tts.TextToSpeech.ERROR;

import android.content.Intent;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.widget.ImageButton;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.examples.shownow1.R;

import java.util.Locale;

public class TutorialActivity extends AppCompatActivity {
    private TextToSpeech tts;  // TTS 변수 선언

    private ImageButton btn_two, btn_four, btn_six, btn_start2;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate (savedInstanceState);
        setContentView (R.layout.activity_tutorial);

        btn_two = (ImageButton)findViewById (R.id.btn_two); // 2배속
        btn_four = (ImageButton)findViewById (R.id.btn_four); // 4배속
        btn_six = (ImageButton)findViewById (R.id.btn_six); // 6배속
        btn_start2 = (ImageButton)findViewById (R.id.btn_start2); // 시작하기


        // TTS를 생성하고 OnInitListener로 초기화 한다.
        tts = new TextToSpeech (this, new TextToSpeech.OnInitListener () {
            @Override
            public void onInit(int status) {
                if (status != ERROR) {
                    // 언어를 선택한다.
                    tts.setLanguage (Locale.KOREAN);
                    tts.setPitch (1.0f);
                }
            }
        });

        // TTS 버튼 클릭 리스너 설정

        btn_two.setOnClickListener (v -> {
            String text = getResources ().getString (R.string.tutorial_msg);
            tts.setSpeechRate (2.0f);
            tts.speak (text, TextToSpeech.QUEUE_FLUSH, null);
        });


        btn_four.setOnClickListener (v -> {
            String text = getResources ().getString (R.string.tutorial_msg);
            tts.setSpeechRate (4.0f);
            tts.speak (text, TextToSpeech.QUEUE_FLUSH, null);
        });


        btn_six.setOnClickListener (v -> {
            String text = getResources ().getString (R.string.tutorial_msg);
            tts.setSpeechRate (6.0f);
            tts.speak (text, TextToSpeech.QUEUE_FLUSH, null);
        });


        btn_start2.setOnClickListener (view -> {
            onDestroy ();
            Intent intent = new Intent (TutorialActivity.this, DetectorActivity.class);
            startActivity (intent); // DetectorActivity로 이동
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy ();
        // TTS 객체가 남아있다면 실행을 중지하고 메모리에서 제거한다.
        if (tts != null) {
            tts.stop ();
            tts.shutdown ();
            tts = null;
        }
    }
}