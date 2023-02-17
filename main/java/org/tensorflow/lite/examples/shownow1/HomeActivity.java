package org.tensorflow.lite.examples.shownow1;

import android.content.Intent;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.widget.ImageButton;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.examples.shownow1.R;

import java.util.Locale;

public class HomeActivity extends AppCompatActivity {
    private ImageButton btn_tutorial;
    private ImageButton btn_start;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home);

        //튜토리얼 듣기
        btn_tutorial = (ImageButton) findViewById(R.id.btn_tutorial);
        btn_tutorial.setOnClickListener(view -> {
            Intent intent = new Intent(getApplicationContext(), TutorialActivity.class);
            startActivity(intent); // 액티비티 이동
        });

        //시작하기
        btn_start = (ImageButton) findViewById(R.id.btn_start);
        btn_start.setOnClickListener(view -> {
            Intent intent = new Intent(getApplicationContext(), DetectorActivity.class);
            startActivity(intent); // 액티비티 이동
        });
    }

}