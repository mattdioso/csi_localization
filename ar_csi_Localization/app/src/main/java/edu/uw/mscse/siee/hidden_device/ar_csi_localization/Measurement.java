package edu.uw.mscse.siee.hidden_device.ar_csi_localization;

import android.app.Activity;
import android.app.ActivityManager;
import android.app.AlertDialog;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.res.Resources;
import android.graphics.Color;
import android.media.Image;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.NotificationCompat;
import androidx.fragment.app.FragmentManager;

import com.google.ar.core.Anchor;
import com.google.ar.core.Frame;
import com.google.ar.core.HitResult;
import com.google.ar.core.Plane;
import com.google.ar.core.Pose;
import com.google.ar.sceneform.AnchorNode;
import com.google.ar.sceneform.FrameTime;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.Scene;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.MaterialFactory;
import com.google.ar.sceneform.rendering.ModelRenderable;
import com.google.ar.sceneform.rendering.Renderable;
import com.google.ar.sceneform.rendering.ShapeFactory;
import com.google.ar.sceneform.rendering.ViewRenderable;
import com.google.ar.sceneform.ux.ArFragment;
import com.google.ar.sceneform.ux.TransformableNode;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Measurement extends AppCompatActivity implements Scene.OnUpdateListener {

    private double MIN_OPENGL_VERSION = 3.0;
    private String TAG = Measurement.class.getSimpleName();

    private ArFragment arFragment = null;
    private TextView distanceModeTextView = null;
    private TextView pointTextView;

    private LinearLayout arrow1UpLinearLayout;
    private LinearLayout arrow1DownLinearLayout;
    private ImageView arrow1UpView;
    private ImageView arrow1DownView;
    private Renderable arrow1UpRenderable;
    private Renderable arrow1DownRenderable;

    private LinearLayout arrow10UpLinearLayout;
    private LinearLayout arrow10DownLinearLayout;
    private ImageView arrow10UpView;
    private ImageView arrow10DownView;
    private Renderable arrow10UpRenderable;
    private Renderable arrow10DownRenderable;

    private TableLayout multipleDistanceTableLayout;
    private ModelRenderable cubeRenderable;
    private ViewRenderable distanceCardViewRenderable;

    private Spinner distanceModeSpinner;
    private ArrayList<String> distanceModeArrayList = new ArrayList<>();
    private String distanceMode = "";

    private ArrayList<Anchor> placedAnchors = new ArrayList<>();
    private ArrayList<AnchorNode> placedAnchorNodes = new ArrayList<>();
    private HashMap<String, Anchor> midAnchors = new HashMap<>();
    private HashMap<String, AnchorNode> midAnchorNodes = new HashMap<>();
    private ArrayList<List<Node>> fromGroundNodes = new ArrayList<>();

    private TextView multipleDistances[][] = new TextView[Constants.maxNumMultiplePoints][Constants.maxNumMultiplePoints];
    private String initCM;
    private Button clearButton;
    private Button calculateButton;
    private String NOTIFICATION_CHANNEL = "AR_CSI_CHANNEL";

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Resources resources = getResources();
        FragmentManager supportFragmentManager = getSupportFragmentManager();

        super.onCreate(savedInstanceState);
        if (!checkIsSupportedDeviceOrFinish(this)) {
            Toast.makeText(this, "Device not supported", Toast.LENGTH_LONG).show();
        }

        setContentView(R.layout.activity_main);
        String[] distanceModeArray = resources.getStringArray(R.array.distance_mode);
        for (int i = 0; i < distanceModeArray.length; i++) {
            distanceModeArrayList.add(distanceModeArray[i]);
        }

        arFragment = (ArFragment) supportFragmentManager.findFragmentById(R.id.sceneform_fragment);
        distanceModeTextView = findViewById(R.id.distance_view);
        multipleDistanceTableLayout = findViewById(R.id.multiple_distance_table);

        initCM = resources.getString(R.string.initCM);

        //configureSpinner();
        //configureView();
        initArrowView();
        initRenderable();
        calculateButton();
        clearButton();

        arFragment.setOnTapArPlaneListener((HitResult hitResult, Plane plane, MotionEvent motionEvent) -> {
            if (cubeRenderable == null || distanceCardViewRenderable == null) return;
                tapDistanceOfMultiplePoints(hitResult);
//            switch(distanceMode) {
//                case "Distance from camera":
//                    clearAllAnchors();
//                    placeAnchor(hitResult, distanceCardViewRenderable);
//                    break;
//                case "Distance of 2 points":
//                    tapDistanceOf2Points(hitResult);
//                    break;
//                case "Distance of multiple points":
//                    tapDistanceOfMultiplePoints(hitResult);
//                    break;
//                case "Distance from ground":
//                    tapDistanceFromGround(hitResult);
//                    break;
//                default:
//                    clearAllAnchors();
//                    placeAnchor(hitResult, distanceCardViewRenderable);
//                    break;
//            }
        });
    }

    @Override
    public void onUpdate(FrameTime frameTime) {

        switch(distanceMode) {
            case "Distance from camera":
                measureDistanceFromCamera();
                break;
            case "Distance of 2 points":
                measureDistanceOf2Points();
                break;
            case "Distance of multiple points":
                Log.i(TAG, "multiple");
                measureMultipleDistances();
                break;
            case "Distance from ground":
                measureDistanceFromGround();
                break;
            default:
                measureDistanceFromCamera();
                break;
        }
    }

    private void initDistanceTable() {
        for (int i = 0; i < Constants.maxNumMultiplePoints + 1; i++) {
            TableRow tableRow = new TableRow(this);
            multipleDistanceTableLayout.addView(tableRow, multipleDistanceTableLayout.getWidth(), Constants.multipleDistanceTableHeight / (Constants.maxNumMultiplePoints + 1));
            for (int j = 0; j < Constants.maxNumMultiplePoints + 1; j++) {
                TextView textView = new TextView(this);
                textView.setTextColor(Color.WHITE);
                if (i == 0) {
                    if (j == 0) {
                        textView.setText("cm");
                    } else {
                        textView.setText(String.valueOf(j-1));
                    }
                } else {
                    if (j == 0) {
                        textView.setText(String.valueOf(i-1));
                    } else if (i == j) {
                        textView.setText("-");
                        multipleDistances[i-1][j-1] = textView;
                    } else {
                        textView.setText(initCM);
                        multipleDistances[i-1][j-1] = textView;
                    }
                }
                tableRow.addView(textView,
                        tableRow.getLayoutParams().width / (Constants.maxNumMultiplePoints + 1),
                        tableRow.getLayoutParams().height);
            }
        }
    }

    private void initArrowView() {
        arrow1UpLinearLayout = new LinearLayout(this);
        arrow1UpLinearLayout.setOrientation(LinearLayout.VERTICAL);
        arrow1UpLinearLayout.setGravity(Gravity.CENTER);
        arrow1UpView = new ImageView(this);
        arrow1UpView.setImageResource(R.drawable.arrow_1up);
        arrow1UpLinearLayout.addView(arrow1UpView, Constants.arrowViewSize, Constants.arrowViewSize);

        arrow1DownLinearLayout = new LinearLayout(this);
        arrow1DownLinearLayout.setOrientation(LinearLayout.VERTICAL);
        arrow1DownLinearLayout.setGravity(Gravity.CENTER);
        arrow1DownView = new ImageView(this);
        arrow1DownView.setImageResource(R.drawable.arrow_1down);
        arrow1DownLinearLayout.addView(arrow1DownView, Constants.arrowViewSize, Constants.arrowViewSize);

        arrow10UpLinearLayout = new LinearLayout(this);
        arrow10UpLinearLayout.setOrientation(LinearLayout.VERTICAL);
        arrow10UpLinearLayout.setGravity(Gravity.CENTER);
        arrow10UpView = new ImageView(this);
        arrow10UpView.setImageResource(R.drawable.arrow_10up);
        arrow10UpLinearLayout.addView(arrow10UpView, Constants.arrowViewSize, Constants.arrowViewSize);

        arrow10DownLinearLayout = new LinearLayout(this);
        arrow10DownLinearLayout.setOrientation(LinearLayout.VERTICAL);
        arrow10DownLinearLayout.setGravity(Gravity.CENTER);
        arrow10DownView = new ImageView(this);
        arrow10DownView.setImageResource(R.drawable.arrow_10down);
        arrow10DownLinearLayout.addView(arrow10DownView, Constants.arrowViewSize, Constants.arrowViewSize);
    }

    private void initRenderable() {

        MaterialFactory.makeTransparentWithColor(this,
                new com.google.ar.sceneform.rendering.Color(Color.RED))
                .thenAccept(material -> {
                    cubeRenderable = ShapeFactory.makeSphere(0.02f,
                            Vector3.zero(),
                            material);
                    cubeRenderable.setShadowCaster(false);
                    cubeRenderable.setShadowReceiver(false);
                })
                .exceptionally(ex -> {
                            AlertDialog.Builder builder = new AlertDialog.Builder(this);
                            builder.setMessage(ex.getMessage()).setTitle("Error");
                            AlertDialog dialog = builder.create();
                            dialog.show();
                            return null;
                        }

                );

        ViewRenderable.builder()
                .setView(getApplicationContext(), R.layout.distance_text_layout)
                .build()
                .thenAccept(viewRenderable -> {
                    distanceCardViewRenderable = viewRenderable;
                    distanceCardViewRenderable.setShadowCaster(false);
                    distanceCardViewRenderable.setShadowReceiver(false);
                })
                .exceptionally(ex -> {
                    AlertDialog.Builder builder = new AlertDialog.Builder(this);
                    builder.setMessage(ex.getMessage()).setTitle("Error");
                    AlertDialog dialog = builder.create();
                    dialog.show();
                    return null;
                });

        ViewRenderable.builder()
                .setView(this, arrow1UpLinearLayout)
                .build()
                .thenAccept(viewRenderable -> {
                    arrow1UpRenderable = viewRenderable;
                    arrow1UpRenderable.setShadowCaster(false);
                    arrow1UpRenderable.setShadowReceiver(false);
                })
                .exceptionally(ex -> {
                    AlertDialog.Builder builder = new AlertDialog.Builder(this);
                    builder.setMessage(ex.getMessage()).setTitle("Error");
                    AlertDialog dialog = builder.create();
                    dialog.show();
                    return null;
                });

        ViewRenderable.builder()
                .setView(this, arrow10UpLinearLayout)
                .build()
                .thenAccept(viewRenderable -> {
                    arrow10UpRenderable = viewRenderable;
                    arrow10UpRenderable.setShadowCaster(false);
                    arrow10UpRenderable.setShadowReceiver(false);
                })
                .exceptionally(ex -> {
                    AlertDialog.Builder builder = new AlertDialog.Builder(this);
                    builder.setMessage(ex.getMessage()).setTitle("Error");
                    AlertDialog dialog = builder.create();
                    dialog.show();
                    return null;
                });

        ViewRenderable.builder()
                .setView(this, arrow10DownLinearLayout)
                .build()
                .thenAccept(viewRenderable -> {
                    arrow10DownRenderable = viewRenderable;
                    arrow10DownRenderable.setShadowReceiver(false);
                    arrow10DownRenderable.setShadowCaster(false);
                })
                .exceptionally(ex -> {
                    AlertDialog.Builder builder = new AlertDialog.Builder(this);
                    builder.setMessage(ex.getMessage()).setTitle("Error");
                    AlertDialog dialog = builder.create();
                    dialog.show();
                    return null;
                });
    }

    private void clearButton() {
        clearButton = findViewById(R.id.clearButton);
        clearButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                clearAllAnchors();
            }
        });
    }

    private void calculateButton() {
        calculateButton = findViewById(R.id.calculateButton);
        calculateButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                calculateVolume();
            }
        });
    }

    private void calculateVolume() {
        if (placedAnchorNodes.size() != 4) {
            displayAlert();
            return;
        }
        AnchorNode vertex = placedAnchorNodes.get(0);
        float l = changeUnit(calculateDistance(vertex.getWorldPosition(), placedAnchorNodes.get(1).getWorldPosition()), "cm");
        float w = changeUnit(calculateDistance(vertex.getWorldPosition(), placedAnchorNodes.get(2).getWorldPosition()), "cm");
        float h = changeUnit(calculateDistance(vertex.getWorldPosition(), placedAnchorNodes.get(3).getWorldPosition()), "cm");
        float vol = l * w * h;
        displayVolume(vol, l, w, h);
        Log.i(TAG, "VOLUME: " + vol);
    }

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            CharSequence name = NOTIFICATION_CHANNEL;
            String description = "notification channel for ar_csi_localization";
            int importance = NotificationManager.IMPORTANCE_DEFAULT;
            NotificationChannel channel = new NotificationChannel(NOTIFICATION_CHANNEL, name, importance);
            channel.setDescription(description);
            NotificationManager notificationManager = getSystemService(NotificationManager.class);
            notificationManager.createNotificationChannel(channel);
        }
    }

    public void displayVolume(float vol, float l, float w, float h) {
        AlertDialog.Builder builder = new AlertDialog.Builder(Measurement.this);
//        float volCM = changeUnit(vol, "cm");
//        float lCM = changeUnit(l, "cm");
//        float wCM = changeUnit(w, "cm");
//        float hCM = changeUnit(h, "cm");
        builder.setMessage("VOLUME: " + vol + " cm\n" + "LENGTH: " + l + " cm\n" + "WIDTH: " + w + " cm\n" + "HEIGHT: " + h + " cm");
        builder.setTitle("Measured Volume");
        builder.setCancelable(true);
        builder.setPositiveButton("Ok", (DialogInterface.OnClickListener) (dialog, which) -> {
            dialog.dismiss();
        });
        AlertDialog alertDialog = builder.create();
        alertDialog.show();
    }

    public void displayAlert() {
        AlertDialog.Builder builder = new AlertDialog.Builder(Measurement.this);
        builder.setMessage("You need to set 4 anchors to be able to calculate the volume");
        builder.setTitle("Not enough Anchors");
        builder.setCancelable(true);
        builder.setPositiveButton("Ok", (DialogInterface.OnClickListener) (dialog, which) -> {
            dialog.dismiss();
        });

        AlertDialog alertDialog = builder.create();
        alertDialog.show();
    }

    private void configureView() {
        clearAllAnchors();
        toastMode();
        distanceModeTextView.setText(distanceModeArrayList.get(0));
        distanceMode = distanceModeArrayList.get(0);
        ViewGroup.LayoutParams layoutParams = multipleDistanceTableLayout.getLayoutParams();
        layoutParams.height = Constants.multipleDistanceTableHeight;
        multipleDistanceTableLayout.setLayoutParams(layoutParams);
        initDistanceTable();
    }

    private void configureSpinner() {
        distanceMode = distanceModeArrayList.get(0);
//        distanceModeSpinner = findViewById(R.id.distance_mode_spinner);
//        ArrayAdapter<String> distanceModeAdapter = new ArrayAdapter<String>(getApplicationContext(), android.R.layout.simple_spinner_item, distanceModeArrayList);
////        clearAllAnchors();
////        toastMode();
////        ViewGroup.LayoutParams layoutParams = multipleDistanceTableLayout.getLayoutParams();
////        layoutParams.height = Constants.multipleDistanceTableHeight;
////        multipleDistanceTableLayout.setLayoutParams(layoutParams);
////        initDistanceTable();
//        distanceModeAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
//        distanceModeSpinner.setAdapter(distanceModeAdapter);
//        distanceModeSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
//            @Override
//            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
//                Spinner spinnerParent = (Spinner)parent;
//                distanceMode = (String) spinnerParent.getSelectedItem();
//                clearAllAnchors();
//                setMode();
//                toastMode();
////                if (distanceMode.equals(distanceModeArrayList.get(2))) {
//                    ViewGroup.LayoutParams layoutParams = multipleDistanceTableLayout.getLayoutParams();
//                    layoutParams.height = Constants.multipleDistanceTableHeight;
//                    multipleDistanceTableLayout.setLayoutParams(layoutParams);
//                    initDistanceTable();
////                } else {
////                    ViewGroup.LayoutParams layoutParams = multipleDistanceTableLayout.getLayoutParams();
////                    layoutParams.height = 0;
////                    multipleDistanceTableLayout.setLayoutParams(layoutParams);
////                }
//                Log.i(TAG, "Selected arcore focus on " + distanceMode);
//            }
//
//            @Override
//            public void onNothingSelected(AdapterView<?> parent) {
//                clearAllAnchors();
//                setMode();
//                toastMode();
//            }
//        });
    }

    private void setMode() {
        Log.i(TAG, distanceMode);
        distanceModeTextView.setText(distanceMode);
    }

    private void clearAllAnchors() {
        placedAnchors.clear();
        for (AnchorNode node : placedAnchorNodes) {
            arFragment.getArSceneView().getScene().removeChild(node);
            node.setEnabled(false);
            node.getAnchor().detach();
            node.setParent(null);
        }
        placedAnchorNodes.clear();
        midAnchors.clear();
        for (Map.Entry<String, AnchorNode> entry : midAnchorNodes.entrySet()) {
            AnchorNode node = entry.getValue();
            arFragment.getArSceneView().getScene().removeChild(node);
            node.setEnabled(false);
            node.getAnchor().detach();
            node.setParent(null);
        }
        midAnchorNodes.clear();
        Measurement.this.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                for (int i = 0; i < Constants.maxNumMultiplePoints; i++) {
                    for (int j = 0; j < Constants.maxNumMultiplePoints; j++) {
                        if (multipleDistances[i][j] != null) {

                            String text = (i == j ? "-" : initCM);
                            //multipleDistances[i][j] = new TextView(this);
                            multipleDistances[i][j].setText(text);
                            //multipleDistances[i][j].invalidate();

                            //findViewById(R.id.multiple_distance_table).invalidate();

                            Log.i(TAG, multipleDistances[i][j].getText().toString());
                        }

                    }
                }
                multipleDistanceTableLayout.invalidate();
            }
        });
        //multipleDistanceTableLayout.invalidate();
        fromGroundNodes.clear();
    }

    private void placeAnchor(HitResult hitResult, Renderable renderable) {
        Anchor anchor = hitResult.createAnchor();
        placedAnchors.add(anchor);

        AnchorNode anchorNode = new AnchorNode(anchor);
        anchorNode.setSmoothed(true);
        anchorNode.setParent(arFragment.getArSceneView().getScene());

        placedAnchorNodes.add(anchorNode);

        TransformableNode node = new TransformableNode(arFragment.getTransformationSystem());
        node.getRotationController().setEnabled(false);
        node.getScaleController().setEnabled(false);
        node.getTranslationController().setEnabled(true);
        node.setRenderable(renderable);
        node.setParent(anchorNode);

        arFragment.getArSceneView().getScene().addOnUpdateListener(this);
        arFragment.getArSceneView().getScene().addChild(anchorNode);
        node.select();
    }

    private void tapDistanceOf2Points(HitResult hitResult) {
        if (placedAnchorNodes.size() == 0)
            placeAnchor(hitResult, cubeRenderable);
        else if (placedAnchorNodes.size() == 1) {
            placeAnchor(hitResult, cubeRenderable);
            float[] midPosition = {
                    (placedAnchorNodes.get(0).getWorldPosition().x + placedAnchorNodes.get(1).getWorldPosition().x) / 2,
                    (placedAnchorNodes.get(0).getWorldPosition().y + placedAnchorNodes.get(1).getWorldPosition().y) / 2,
                    (placedAnchorNodes.get(0).getWorldPosition().z + placedAnchorNodes.get(1).getWorldPosition().z) / 2
            };

            float [] quaternion = { 0.0f, 0.0f, 0.0f, 0.0f };
            Pose pose = new Pose(midPosition, quaternion);

            placeMidAnchor(pose, distanceCardViewRenderable);
        } else {
            clearAllAnchors();
            placeAnchor(hitResult, cubeRenderable);
        }
    }

    private void tapDistanceOfMultiplePoints(HitResult hitResult) {
        if (placedAnchorNodes.size() >= Constants.maxNumMultiplePoints)
            clearAllAnchors();

        ViewRenderable.builder()
                .setView(this, R.layout.point_text_layout)
                .build()
                .thenAccept(viewRenderable -> {
                    viewRenderable.setShadowReceiver(false);
                    viewRenderable.setShadowCaster(false);
                    pointTextView = (TextView)viewRenderable.getView();
                    pointTextView.setText(String.valueOf(placedAnchors.size()));
                    placeAnchor(hitResult, viewRenderable);
                })
                .exceptionally(ex -> {
                    AlertDialog.Builder builder = new AlertDialog.Builder(this);
                    builder.setMessage(ex.getMessage()).setTitle("Error");
                    AlertDialog dialog = builder.create();
                    dialog.show();
                    return null;
                });
        Log.i(TAG, "Number of anchors: "+ placedAnchorNodes.size());
    }

    private void tapDistanceFromGround(HitResult hitResult) {
        clearAllAnchors();
        Anchor anchor = hitResult.createAnchor();
        placedAnchors.add(anchor);

        AnchorNode anchorNode = new AnchorNode(anchor);
        anchorNode.setSmoothed(true);
        anchorNode.setParent(arFragment.getArSceneView().getScene());
        placedAnchorNodes.add(anchorNode);

        TransformableNode transformableNode = new TransformableNode(arFragment.getTransformationSystem());
        transformableNode.getRotationController().setEnabled(false);
        transformableNode.getScaleController().setEnabled(false);
        transformableNode.setRenderable(cubeRenderable);
        transformableNode.setParent(anchorNode);

        Node node = new Node();
        node.setParent(transformableNode);
        node.setWorldPosition(new Vector3(anchorNode.getWorldPosition().x,
                anchorNode.getWorldPosition().y,
                anchorNode.getWorldPosition().z));
        node.setRenderable(distanceCardViewRenderable);

        Node arrow1UpNode = new Node();
        arrow1UpNode.setParent(node);
        arrow1UpNode.setWorldPosition(new Vector3(
                node.getWorldPosition().x,
                node.getWorldPosition().y + 0.1f,
                node.getWorldPosition().z
        ));
        arrow1UpNode.setRenderable(arrow1UpRenderable);
        arrow1UpNode.setOnTapListener((HitTestResult htr, MotionEvent me) -> {
            node.setWorldPosition(new Vector3(
                    node.getWorldPosition().x,
                    node.getWorldPosition().y + 0.01f,
                    node.getWorldPosition().z
            ));
        });

        Node arrow1DownNode = new Node();
        arrow1DownNode.setParent(node);
        arrow1DownNode.setWorldPosition(new Vector3(
                node.getWorldPosition().x,
                node.getWorldPosition().y - 0.08f,
                node.getWorldPosition().z
        ));
        arrow1DownNode.setRenderable(arrow1DownRenderable);
        arrow1DownNode.setOnTapListener((HitTestResult htr, MotionEvent me) -> {
            node.setWorldPosition(new Vector3(
                    node.getWorldPosition().x,
                    node.getWorldRotation().y - 0.01f,
                    node.getWorldPosition().z
            ));
        });

        Node arrow10UpNode = new Node();
        arrow10UpNode.setParent(node);
        arrow10UpNode.setWorldPosition(new Vector3(
                node.getWorldPosition().x,
                node.getWorldPosition().y + 0.18f,
                node.getWorldPosition().z
        ));
        arrow10UpNode.setRenderable(arrow10UpRenderable);
        arrow10UpNode.setOnTapListener((HitTestResult htr, MotionEvent me) -> {
            node.setWorldPosition(new Vector3(
                    node.getWorldPosition().x,
                    node.getWorldPosition().y + 0.1f,
                    node.getWorldPosition().z
            ));
        });

        Node arrow10DownNode = new Node();
        arrow10DownNode.setParent(node);
        arrow10DownNode.setWorldPosition(new Vector3(
                node.getWorldPosition().x,
                node.getWorldPosition().y - 0.167f,
                node.getWorldPosition().z
        ));
        arrow10DownNode.setRenderable(arrow10DownRenderable);
        arrow10DownNode.setOnTapListener((HitTestResult htr, MotionEvent me) -> {
            node.setWorldPosition(new Vector3(
                    node.getWorldPosition().x,
                    node.getWorldPosition().y - 0.1f,
                    node.getWorldPosition().z
            ));
        });
        List<Node> items = new ArrayList<Node>() {{
            add(node);
            add(arrow1UpNode);
            add(arrow1DownNode);
            add(arrow10UpNode);
            add(arrow10DownNode);
        }};
        fromGroundNodes.add(items);

        arFragment.getArSceneView().getScene().addOnUpdateListener(this);
        arFragment.getArSceneView().getScene().addChild(anchorNode);
        transformableNode.select();
    }

    private void placeMidAnchor(Pose pose, Renderable renderable) {
        int [] between = {0, 1};
        String mid_key = between[0] + "_" + between[1];
        Anchor anchor = arFragment.getArSceneView().getSession().createAnchor(pose);
        midAnchors.put(mid_key, anchor);

        AnchorNode anchorNode = new AnchorNode(anchor);
        anchorNode.setSmoothed(true);
        anchorNode.setParent(arFragment.getArSceneView().getScene());
        midAnchorNodes.put(mid_key, anchorNode);

        TransformableNode node = new TransformableNode(arFragment.getTransformationSystem());
        node.getRotationController().setEnabled(false);
        node.getScaleController().setEnabled(false);
        node.getTranslationController().setEnabled(true);
        node.setRenderable(renderable);
        node.setParent(anchorNode);

        arFragment.getArSceneView().getScene().addOnUpdateListener(this);
        arFragment.getArSceneView().getScene().addChild(anchorNode);
    }

    private void measureDistanceFromGround() {
        if (fromGroundNodes.size() == 0) return;
        for (List node : fromGroundNodes) {
            TextView textView = (TextView)((LinearLayout) distanceCardViewRenderable.getView()).findViewById(R.id.distanceCard);
            Node node1 = (Node)node.get(0);
            float distanceCM = changeUnit(node1.getWorldPosition().y + 1.0f, "cm");
            textView.setText(distanceCM + " cm");
        }
    }

    private void measureDistanceFromCamera() {
        Frame frame = arFragment.getArSceneView().getArFrame();
        if (placedAnchorNodes.size() >= 1) {
            float distanceMeter = calculateDistance(placedAnchorNodes.get(0).getWorldPosition(), frame.getCamera().getPose());
            measureDistanceOf2Points(distanceMeter);
        }
    }

    private void measureDistanceOf2Points() {
        if (placedAnchorNodes.size() == 2) {
            float distanceMeter = calculateDistance(placedAnchorNodes.get(0).getWorldPosition(),
                    placedAnchorNodes.get(1).getWorldPosition());
            measureDistanceOf2Points(distanceMeter);
        }
    }

    private void measureMultipleDistances() {
        if (placedAnchorNodes.size() > 1) {
            for (int i = 0; i < placedAnchorNodes.size(); i++) {
                for (int j = i + 1; j < placedAnchorNodes.size(); j++) {
                    float distanceMeter = calculateDistance(
                            placedAnchorNodes.get(i).getWorldPosition(),
                            placedAnchorNodes.get(j).getWorldPosition()
                    );
                    float distanceCM = changeUnit(distanceMeter, "cm");
                    String distanceCMFloor = "%.2f".format(String.valueOf(distanceCM));
                    Log.d(TAG, "%.2f".format(String.valueOf(distanceCM)));
                    multipleDistances[i][j].setText(distanceCMFloor);
                    multipleDistances[j][i].setText(distanceCMFloor);
                }
            }
        }
    }

    private void measureDistanceOf2Points(float distanceMeter) {
        String distanceTextCM = makeDistanceTextWithCM(distanceMeter);
        TextView textView = (TextView)((LinearLayout) distanceCardViewRenderable.getView()).findViewById(R.id.distanceCard);
        textView.setText(distanceTextCM);
        //Log.d(TAG, "distance: " + distanceTextCM);
    }

    private String makeDistanceTextWithCM(float distanceMeter) {
        float distanceCM = changeUnit(distanceMeter, "cm");
        String distanceCMFloor = "%.2f".format(String.valueOf(distanceCM));
        return distanceCMFloor + " cm";
    }

    private float calculateDistance(float x, float y, float z) {
        return (float)Math.sqrt(Math.pow(x, 2) + Math.pow(y, 2) + Math.pow(z, 2));
    }

    private float calculateDistance(Pose objectPose0, Pose objectPose1) {
        return calculateDistance(
                objectPose0.tx() - objectPose1.tx(),
                objectPose0.ty() - objectPose1.ty(),
                objectPose0.tz() - objectPose1.tz()
        );
    }

    private float calculateDistance(Vector3 objectPose0, Pose objectPose1) {
        return calculateDistance(
                objectPose0.x - objectPose1.tx(),
                objectPose0.y - objectPose1.ty(),
                objectPose0.z - objectPose1.tz()
        );
    }

    private float calculateDistance(Vector3 objectPose0, Vector3 objectPose1) {
        return calculateDistance(
                objectPose0.x - objectPose1.x,
                objectPose0.y - objectPose1.y,
                objectPose0.z - objectPose1.z
        );
    }

    private float changeUnit(float distanceMeter, String unit) {
        float ret = 0.0f;
        switch(unit) {
            case "cm":
                ret = distanceMeter * 100;
                break;
            case "mm":
                ret = distanceMeter * 1000;
                break;
            default:
                ret = distanceMeter;
        }
        return ret;
    }

    private boolean checkIsSupportedDeviceOrFinish(Activity activity) {
        String openGlVersionString = ((ActivityManager) activity.getSystemService(Context.ACTIVITY_SERVICE)).getDeviceConfigurationInfo().getGlEsVersion();
        if (Double.parseDouble(openGlVersionString) < MIN_OPENGL_VERSION) {
            Log.e(TAG, "Sceneform requires OpenGL ES " + MIN_OPENGL_VERSION + " or later");
            Toast.makeText(activity, "Sceneform requires OpenGL ES " + MIN_OPENGL_VERSION + " or later", Toast.LENGTH_LONG).show();
            activity.finish();
            return false;
        }
        return true;
    }

    private void toastMode() {
        Toast.makeText(this, "", Toast.LENGTH_LONG).show();
    }
}
