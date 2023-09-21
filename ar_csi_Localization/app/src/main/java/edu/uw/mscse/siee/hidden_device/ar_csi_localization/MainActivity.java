package edu.uw.mscse.siee.hidden_device.ar_csi_localization;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentManager;
import androidx.fragment.app.FragmentOnAttachListener;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;

import com.google.ar.core.Config;
import com.google.ar.core.HitResult;
import com.google.ar.core.Plane;
import com.google.ar.core.Session;
import com.google.ar.sceneform.ArSceneView;
import com.google.ar.sceneform.SceneView;
import com.google.ar.sceneform.ux.ArFragment;
import com.google.ar.sceneform.ux.BaseArFragment;

import java.util.ArrayList;

public class MainActivity extends AppCompatActivity implements
        FragmentOnAttachListener,
        BaseArFragment.OnTapArPlaneListener,
        BaseArFragment.OnSessionConfigurationListener {


    private static final String TAG = MainActivity.class.getSimpleName();
    private final ArrayList<String> buttonArrayList = new ArrayList<>();
    private Button toMeasurement;

    private ArFragment arFragment;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_measurement);
        String[] buttonArray = getResources().getStringArray(R.array.arcore_measurement_buttons);
        for (int i = 0; i < buttonArray.length ; i++) {
            Log.i(TAG, buttonArray[i]);
            buttonArrayList.add(buttonArray[i]);
        }

        toMeasurement = findViewById(R.id.to_measurement);
        toMeasurement.setText(buttonArrayList.get(0));
        toMeasurement.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getApplication(), Measurement.class);
                startActivity(intent);
            }
        });
    }

    @Override
    public void onAttachFragment(@NonNull FragmentManager fragmentManager, @NonNull Fragment fragment) {
        arFragment = (ArFragment) fragment;
        arFragment.setOnSessionConfigurationListener(this);
        //arFragment.setOnViewCreatedListener(this);
        arFragment.setOnTapArPlaneListener(this);
    }

    @Override
    public void onSessionConfiguration(Session session, Config config) {
        if (session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
            config.setDepthMode(Config.DepthMode.AUTOMATIC);
        }
    }

//    @Override
//    public void onViewCreated(ArSceneView arSceneView) {
//        arFragment.setOnViewCreatedListener(null);
//        arSceneView.setFrameRateFactor(SceneView.FrameRate.FULL);
//    }

    @Override
    public void onTapPlane(HitResult hitResult, Plane plance, MotionEvent motionEvent) {

    }
}