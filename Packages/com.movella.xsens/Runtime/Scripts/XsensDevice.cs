using UnityEngine;
using Unity.LiveCapture;
using Unity.LiveCapture.Mocap;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine.Playables;

namespace Movella.Xsens
{
    [CreateDeviceMenuItem("Xsens Actor Device")]
    [AddComponentMenu("MVN/Xsens Actor Device")]
    class XsensDevice : MocapDevice<FrameData>
    {
        [SerializeField]
        int m_CharacterID;

        [SerializeField, Tooltip("The global channels of mocap data to apply to this source.")]
        [EnumFlagButtonGroup(60f)]
        JointFlags m_Channels = JointFlags.Position | JointFlags.Rotation;

        [Serializable]
        struct PropInfo
        {
            public GameObject gameObject;
            public XsBodyAnimationSegment segmentType;
        }

        [SerializeField]
        PropInfo[] m_Props = new PropInfo[XsensConstants.MvnPropSegmentCount];

        PropRecorder[] m_PropRecorders;

        [SerializeField, HideInInspector]
        Avatar m_AvatarCache;

        XsensConnection m_Connection;

        bool m_ClientInitialized;
        bool m_ModelInitialized;

        (Transform transform, Vector3 tposePosition, Quaternion tPoseRotation)[] m_Model;  // Model segments with TPose rotation
        (Transform transform, Quaternion rot)[] m_OriginalRotations;

        // If we get no data from the client in the current frame, use the previous frame to avoid hiccups
        FrameData m_PreviousFrame;

        double? m_FirstFrameTime = null;

        int[] m_BodySegmentOrder;
        int[] m_FingerSegmentOrder;

        public override bool IsReady() => Animator != null && IsConnected;

        public bool IsConnected => Client?.IsConnected ?? false;

        public int CharacterID => m_CharacterID;


        // [START] Our code
            public bool start; // Flag to start providing feedback (True to start)
            public int activityNumber; // Activity identified (1, 2, 31, 32)
            public double participantHeight; // Participant's height
            private double stdDeviationQuantity; // Scale for standard deviation (e.g. if std is 0.3 and the scale 2.0, then the std will be scaled as 0.3 * 2.0)
            public bool isRightPerformance; // Flag indicating if it's the right side performing (True) or not (in case of ACT2 or ACT31, it's irrelevant if it's True or False)
            Dictionary<string, Tuple<Tuple<float?, float?>, Tuple<float?, float?>>> thresholdValues; // Dictionary with the reference learning data for each parameter, which will be used to evaluate error conditions (feedback)
            Dictionary<string, Tuple<float?, float?>> maxMinValues; // Dictionary that will be further stored with the maximum and minimum values collected during streaming
            Dictionary<string, List<float>> positionList; // Dictionary that will be further stored with values collected during the streaming of ACT31 and ACT32 (which target tremors)
            DateTime lastTime = DateTime.Now; // Last time reference to measure feedback threshold
            DateTime currentTime; // Current time reference to measure feedback threshold
            float threshold_time = 10.0f; // Default feedback threshold is 10 seconds
            bool errorAct3Rest; // Flag enabled as True if any error occured during ACT31
            bool errorAct1Pos; // Flag enabled as True if any error occured during ACT1 Position
            bool errorAct1Vel; // Flag enabled as True if any error occured during ACT1 Velocity
            bool errorAct2; // Flag enabled as True if any error occured during ACT2
            bool errorAct3Pos; // Flag enabled as True if any error occured during ACT32 Position
            bool errorAct3Vel; // Flag enabled as True if any error occured during ACT32 Velocity
            private AudioClip audioClip; // Audio clip of error message (feedback)
            private AudioSource audioSource; // Audio source for audio clip
            private Queue<AudioClip> audioQueue = new Queue<AudioClip>(); // Queue to stop audios from being played at the same time
            private bool isPlayingAudio = false; // Flag enabled as True if any audio is playing
            private GameObject targetObject; // Game object for the emoji
            private SpriteRenderer targetSpriteRenderer; // Renderer to disable the emoji in case any error occured during performance (by appearing, it means positive feedback)
            private double samplingFrequency = 58.8235;
            private double cutoffFrequency = 3.0;

            // Dictionary with the reference learning data for each parameter of ACT1, which will be used to evaluate error conditions (feedback)
            private Dictionary<string, Tuple<Tuple<float?, float?>, Tuple<float?, float?>>> extremeDeviationValues1 = new Dictionary<string, Tuple<Tuple<float?, float?>, Tuple<float?, float?>>>
            {
                // First tuple refers to reference min value (mean, std) and second tuple refers to reference max value (mean, std)
                {"Position RightFoot y", new Tuple<Tuple<float?, float?>, Tuple<float?, float?>>(new Tuple<float?, float?>((float)0.27, (float)0.04), new Tuple<float?, float?>(null, null))},
                {"Position LeftFoot y", new Tuple<Tuple<float?, float?>, Tuple<float?, float?>>(new Tuple<float?, float?>((float)0.27, (float)0.05), new Tuple<float?, float?>(null, null))},
                {"Velocity RightFoot y", new Tuple<Tuple<float?, float?>, Tuple<float?, float?>>(new Tuple<float?, float?>((float)-0.75, (float)0.21), new Tuple<float?, float?>((float)0.79, (float)0.26))},
                {"Velocity LeftFoot y", new Tuple<Tuple<float?, float?>, Tuple<float?, float?>>(new Tuple<float?, float?>((float)-0.72, (float)0.18), new Tuple<float?, float?>((float)0.79, (float)0.23))}
            };

            // Dictionary with the reference learning data for each parameter of ACT2, which will be used to evaluate error conditions (feedback)
            private Dictionary<string, Tuple<Tuple<float?, float?>, Tuple<float?, float?>>> extremeDeviationValues2 = new Dictionary<string, Tuple<Tuple<float?, float?>, Tuple<float?, float?>>>
            {
                // First tuple refers to reference min value (mean, std) and second tuple refers to reference max value (mean, std)
                { "Position COM z", new Tuple<Tuple<float?, float?>, Tuple<float?, float?>>(new Tuple<float?, float?>((float)-0.13, (float)0.07), new Tuple<float?, float?>((float)-0.1, (float)0.05)) },
                { "Position COM x", new Tuple<Tuple<float?, float?>, Tuple<float?, float?>>(new Tuple<float?, float?>((float)-0.02, (float)0.05), new Tuple<float?, float?>((float)0.18, (float)0.07)) }
            };

            // Dictionary with the reference learning data for each parameter of ACT31, which will be used to evaluate error conditions (feedback)
            private Dictionary<string, Tuple<float, float>> extremeDeviationValues31 = new Dictionary<string, Tuple<float, float>>
            {
                // Tuple refers to reference value (mean, std)
                {"Rest RightHand z", new Tuple<float, float>((float)0.03, (float)0.04)},
                {"Rest RightHand x", new Tuple<float, float>((float)0.05, (float)0.06)},
                {"Rest RightHand y", new Tuple<float, float>((float)0.53, (float)0.05)},
                {"Rest LeftHand z", new Tuple<float, float>((float)-0.22, (float)0.05)},
                {"Rest LeftHand x", new Tuple<float, float>((float)0.04, (float)0.05)},
                {"Rest LeftHand y", new Tuple<float, float>((float)0.52, (float)0.05)}
            };

            // Dictionary with the reference learning data for each parameter of ACT32, which will be used to evaluate error conditions (feedback)
            private Dictionary<string, Tuple<Tuple<float?, float?>, Tuple<float?, float?>>> extremeDeviationValues32 = new Dictionary<string, Tuple<Tuple<float?, float?>, Tuple<float?, float?>>>
            {
                {"Position RightHand z", new Tuple<Tuple<float?, float?>, Tuple<float?, float?>>(new Tuple<float?, float?>((float)-0.05, (float)0.07), new Tuple<float?, float?>((float)-0.01, (float)0.05))},
                {"Position RightHand x", new Tuple<Tuple<float?, float?>, Tuple<float?, float?>>(new Tuple<float?, float?>((float)0.05, (float)0.06), new Tuple<float?, float?>((float)0.23, (float)0.06))},
                {"Position RightHand y", new Tuple<Tuple<float?, float?>, Tuple<float?, float?>>(new Tuple<float?, float?>((float)0.54, (float)0.05), new Tuple<float?, float?>((float)0.60, (float)0.05))},
                {"Position LeftHand z", new Tuple<Tuple<float?, float?>, Tuple<float?, float?>>(new Tuple<float?, float?>((float)-0.19, (float)0.05), new Tuple<float?, float?>((float)-0.16, (float)0.05))},
                {"Position LeftHand x", new Tuple<Tuple<float?, float?>, Tuple<float?, float?>>(new Tuple<float?, float?>((float)0.04, (float)0.06), new Tuple<float?, float?>((float)0.22, (float)0.06))},
                {"Position LeftHand y", new Tuple<Tuple<float?, float?>, Tuple<float?, float?>>(new Tuple<float?, float?>((float)0.53, (float)0.06), new Tuple<float?, float?>((float)0.59, (float)0.05))},
                {"Velocity RightHand x", new Tuple<Tuple<float?, float?>, Tuple<float?, float?>>(new Tuple<float?, float?>((float)-0.28, (float)0.08), new Tuple<float?, float?>((float)0.26, (float)0.08))},
                {"Velocity LeftHand x", new Tuple<Tuple<float?, float?>, Tuple<float?, float?>>(new Tuple<float?, float?>((float)-0.27, (float)0.09), new Tuple<float?, float?>((float)0.26, (float)0.08))}
            };
        // [END] Our code

        XsensClient Client
        {
            get
            {
                if (m_Connection == null)
                    ConnectionManager.Instance.TryGetConnection(out m_Connection);
                return m_Connection?.Client;
            }
        }

        protected override void OnValidate()
        {
            base.OnValidate();
            m_CharacterID = Mathf.Clamp(m_CharacterID, 0, XsensConstants.MaxCharacters - 1);
        }

        protected void Start()
        {
            // [START] Our code
                stdDeviationQuantity = activityNumber switch
                {
                    1 => participantHeight,  // scaled as std * height (for ACT1)
                    2 => participantHeight,  // scaled as std * height (for ACT2)
                    31 => 0.165,             // scaled as std * 0.165  (for ACT31)
                    _ => 1.0                 // no scale (for ACT32)
                };
                targetObject = GameObject.Find("Smile");
                if (targetObject != null)
                {
                    targetSpriteRenderer = targetObject.GetComponent<SpriteRenderer>();
                    if (targetSpriteRenderer == null)
                        targetSpriteRenderer = targetObject.AddComponent<SpriteRenderer>();
                    targetSpriteRenderer.enabled = true;
                }
                lastTime = DateTime.Now;
                audioClip = null;
                audioSource = GetComponent<AudioSource>();
                if (audioSource == null)
                    audioSource = gameObject.AddComponent<AudioSource>();
                audioSource.clip = null;

                thresholdValues = new Dictionary<string, Tuple<Tuple<float?, float?>, Tuple<float?, float?>>>();
                if (activityNumber == 1)
                {
                    thresholdValues = extremeDeviationValues1;
                }
                else if (activityNumber == 32)
                {
                    thresholdValues = extremeDeviationValues32;
                }

                maxMinValues = new Dictionary<string, Tuple<float?, float?>>();
                positionList = new Dictionary<string, List<float>>();
                samplingFrequency = 58.8235;
                cutoffFrequency = 3.0;
                errorAct3Rest = false;
                errorAct1Pos = false;
                errorAct1Vel = false;
                errorAct2 = false;
                errorAct3Pos = false;
                errorAct3Vel = false;
            // [END] Our code
        }

        protected override void OnEnable()
        {
            base.OnEnable();

            if (Animator == null)
                Animator = gameObject.GetOrAddComponent<Animator>();

            // [START] Our code
                stdDeviationQuantity = activityNumber switch
                {
                    1 => participantHeight,
                    2 => participantHeight,
                    31 => 0.165,
                    _ => 1.0
                };

                audioSource = GetComponent<AudioSource>();
                if (audioSource == null)
                    audioSource = gameObject.AddComponent<AudioSource>();

                targetObject = GameObject.Find("Smile");
                if (targetObject != null)
                {
                    targetSpriteRenderer = targetObject.GetComponent<SpriteRenderer>();
                    if (targetSpriteRenderer == null)
                        targetSpriteRenderer = targetObject.AddComponent<SpriteRenderer>();
                    targetSpriteRenderer.enabled = true;
                }

                thresholdValues = new Dictionary<string, Tuple<Tuple<float?, float?>, Tuple<float?, float?>>>();
                if (activityNumber == 1)
                {
                    thresholdValues = extremeDeviationValues1;
                }
                else if (activityNumber == 32)
                {
                    thresholdValues = extremeDeviationValues32;
                }

                maxMinValues = new Dictionary<string, Tuple<float?, float?>>();
                positionList = new Dictionary<string, List<float>>();
                samplingFrequency = 58.8235;
                cutoffFrequency = 3.0;
                errorAct3Rest = false;
                errorAct1Pos = false;
                errorAct1Vel = false;
                errorAct2 = false;
                errorAct3Pos = false;
                errorAct3Vel = false;
            // [END] Our code

            m_ModelInitialized = false;
            m_ClientInitialized = false;

            m_BodySegmentOrder = Enum.GetValues(typeof(XsBodyAnimationSegment)).Cast<int>().ToArray();
            m_FingerSegmentOrder = Enum.GetValues(typeof(XsFingerAnimationSegment)).Cast<int>().ToArray();

            var maxSegments = XsensConstants.MvnBodySegmentCount + XsensConstants.MvnFingerSegmentCount + XsensConstants.MvnPropSegmentCount;

            m_Model = new (Transform, Vector3, Quaternion)[maxSegments];
            m_OriginalRotations = new (Transform, Quaternion)[maxSegments];

            if (Animator != null && Animator.avatar == null && m_AvatarCache != null)
                Animator.avatar = m_AvatarCache;

            m_PropRecorders = new PropRecorder[XsensConstants.MvnPropSegmentCount];

            for (int i = 0; i < m_PropRecorders.Length; i++)
                m_PropRecorders[i] = new PropRecorder();

#if UNITY_EDITOR
            if (!Application.isPlaying)
            {
                UnityEditor.EditorApplication.update -= EditorUpdate;
                UnityEditor.EditorApplication.update += EditorUpdate;
            }
#endif
        }

        protected override void OnDisable()
        {
            base.OnDisable();

            m_ModelInitialized = false;
            m_ClientInitialized = false;

            var client = Client;

            if (client != null)
            {
                client.FrameDataReceivedAsync -= OnFrameDataReceivedAsync;
                client.Disconnected -= OnClientDisconnected;
            }

#if UNITY_EDITOR
            UnityEditor.EditorApplication.update -= EditorUpdate;
#endif
        }

#if UNITY_EDITOR
        void EditorUpdate()
        {
            UnityEditor.EditorApplication.QueuePlayerLoopUpdate();
            UnityEditor.SceneView.RepaintAll();
        }
#endif

        bool InitModel()
        {
            var animator = Animator;

            if (animator == null)
                return false;

            if (animator.avatar == null)
            {
                Debug.LogError($"{nameof(XsensDevice)}: {Animator.name} is missing an avatar. An avatar is required to bind incoming Xsens data to their correct bone destinations.");
                return false;
            }

            for (int i = 0; i < m_Model.Length; ++i)
                m_Model[i] = (null, Vector3.zero, Quaternion.identity);

            RestoreTPose();

            //go through the model's body segments and store values
            for (int i = 0; i < XsensConstants.MvnBodySegmentCount; i++)
            {
                var segID = m_BodySegmentOrder[i];
                HumanBodyBones boneID = XsensConstants.BodyMecAnimBones[(XsBodyAnimationSegment)m_BodySegmentOrder[i]];

                try
                {
                    if (boneID == HumanBodyBones.LastBone)
                        continue;

                    Vector3 tempPos = animator.transform.position;
                    Quaternion tempRot = animator.transform.rotation;

                    animator.transform.position = Vector3.zero;
                    animator.transform.rotation = Quaternion.identity;

                    var bone = animator.GetBoneTransform(boneID);

                    if (bone != null)
                        m_Model[segID] = (bone, bone.position, bone.rotation);

                    animator.transform.position = tempPos;
                    animator.transform.rotation = tempRot;
                }
                catch (Exception e)
                {
                    Debug.Log($"{nameof(XsensDevice)}: Error processing [{boneID}] in the model. Exception: {e}");
                }
            }

            //go through the model's finger segments and store values
            for (int i = 0; i < XsensConstants.MvnFingerSegmentCount; i++)
            {
                var segID = m_FingerSegmentOrder[i] + XsensConstants.MvnBodySegmentCount + XsensConstants.MvnPropSegmentCount;
                HumanBodyBones boneID = XsensConstants.FingerMecAnimBones[(XsFingerAnimationSegment)m_FingerSegmentOrder[i]];

                try
                {
                    if (boneID == HumanBodyBones.LastBone)
                        continue;

                    Vector3 tempPos = animator.transform.position;
                    Quaternion tempRot = animator.transform.rotation;

                    animator.transform.position = Vector3.zero;
                    animator.transform.rotation = Quaternion.identity;

                    var bone = animator.GetBoneTransform(boneID);

                    if (bone != null)
                        m_Model[segID] = (bone, bone.position, bone.rotation);

                    animator.transform.position = tempPos;
                    animator.transform.rotation = tempRot;
                }
                catch (Exception e)
                {
                    Debug.Log($"{nameof(XsensDevice)}: Error processing [{boneID}] in the model. Exception: {e}");
                }
            }

            // go through props and store values
            for (int i = 0; i < XsensConstants.MvnPropSegmentCount; ++i)
            {
                var propID = XsensConstants.MvnBodySegmentCount + i;

                var pinfo = m_Props[i];

                if (pinfo.gameObject != null)
                {
                    var tpose = pinfo.gameObject.GetOrAddComponent<TPose>();
                    tpose.RefreshTPose();
                    m_Model[propID] = (pinfo.gameObject.transform, tpose.Position, tpose.Rotation);
                }
                else
                {
                    m_Model[propID] = (null, Vector3.zero, Quaternion.identity);
                }
            }

            return true;
        }

        // [START] Our code
            /* Updates the minimum and maximum values for bone position, velocity, or COM for a specific body segment. The update is based on the activity number and the relevant body segment ID (segID).

            <param name="segID">The segment ID representing a specific body part.</param>
            <param name="HumanBonesMap">A dictionary mapping segment IDs to human bone names.</param>
            <param name="bone">The position vector of the bone (if received).</param>
            <param name="velocity">The velocity vector of the bone (if received).</param>
            <param name="com">The COM vector (if received).</param> */
            private void UpdateMinMaxValues(int segID, Dictionary<int, string> HumanBonesMap, Vector3? bone, Vector3? velocity, Vector3? com)
            {
                string propertyName = HumanBonesMap[segID];

                // ACT1: Focus on foot movement (right foot, ID: 17, and left foot, ID: 21).
                // For both bone position and velocity, we are only interested in the Y-axis (vertical movement) since the task is to lift the foot.
                if (activityNumber == 1)
                {
                    if (segID == 17 || segID == 21)
                    {
                        // Update bone position (Y-axis).
                        if (bone != null)
                        {
                            float currentPosition = ((Vector3)bone).y;
                            if (!maxMinValues.ContainsKey("Position " + propertyName + " y"))
                                maxMinValues.Add("Position " + propertyName + " y", new Tuple<float?, float?>(currentPosition, null));
                            else
                            {
                                float prevMaxValue = (float)maxMinValues["Position " + propertyName + " y"].Item1;
                                // Update max value if current position exceeds the previously recorded maximum.
                                if (currentPosition > prevMaxValue)
                                    maxMinValues["Position " + propertyName + " y"] = new Tuple<float?, float?>(currentPosition, null);
                            }
                        }
                        // Update bone velocity (Y-axis).
                        if (velocity != null)
                        {
                            float currentVelocity = ((Vector3)velocity).y;
                            if (!maxMinValues.ContainsKey("Velocity " + propertyName + " y"))
                                maxMinValues.Add("Velocity " + propertyName + " y", new Tuple<float?, float?>(currentVelocity, currentVelocity));
                            else
                            {
                                float prevMinVelocity = (float)maxMinValues["Velocity " + propertyName + " y"].Item1;
                                float prevMaxVelocity = (float)maxMinValues["Velocity " + propertyName + " y"].Item2;

                                // Update min and max velocity if current values exceed the previously recorded values.
                                if (currentVelocity < prevMinVelocity)
                                    maxMinValues["Velocity " + propertyName + " y"] = new Tuple<float?, float?>(currentVelocity, prevMaxVelocity);
                                if (currentVelocity > prevMaxVelocity)
                                    maxMinValues["Velocity " + propertyName + " y"] = new Tuple<float?, float?>(prevMinVelocity, currentVelocity);
                            }
                        }
                    }
                }
                else if (activityNumber == 2)
                {
                    // ACT2: Focus on COM for the pelvis (Segment ID: 0).
                    // Only COM data is relevant here, and we track movement in both the Z (mediolateral) and X (anteroposterior) axes.
                    if (segID == 0)
                    {
                        if (com != null)
                        {
                            // Update COM position for Z-axis (mediolateral movement).
                            float currentPositionZ = ((Vector3)com).z;
                            if (!maxMinValues.ContainsKey("Position COM z"))
                                maxMinValues.Add("Position COM z", new Tuple<float?, float?>(currentPositionZ, currentPositionZ));
                            else
                            {
                                float minCOMZ = (float)maxMinValues["Position COM z"].Item1;
                                float maxCOMZ = (float)maxMinValues["Position COM z"].Item2;
                                if (currentPositionZ < minCOMZ)
                                    maxMinValues["Position COM z"] = new Tuple<float?, float?>(currentPositionZ, maxCOMZ);
                                if (currentPositionZ > maxCOMZ)
                                    maxMinValues["Position COM z"] = new Tuple<float?, float?>(minCOMZ, currentPositionZ);
                            }

                            // Update COM position for X-axis (anteroposterior movement).
                            float currentPositionX = ((Vector3)com).x;
                            if (!maxMinValues.ContainsKey("Position COM x"))
                                maxMinValues.Add("Position COM x", new Tuple<float?, float?>(currentPositionX, currentPositionX));
                            else
                            {
                                float minCOMX = (float)maxMinValues["Position COM x"].Item1;
                                float maxCOMX = (float)maxMinValues["Position COM x"].Item2;
                                if (currentPositionX < minCOMX)
                                    maxMinValues["Position COM x"] = new Tuple<float?, float?>(currentPositionX, maxCOMX);
                                if (currentPositionX > maxCOMX)
                                    maxMinValues["Position COM x"] = new Tuple<float?, float?>(minCOMX, currentPositionX);
                            }
                        }
                    }
                }
                else if (activityNumber == 32)
                {
                    // ACT32: Focus on hand movement (right hand, ID: 10, left hand, ID: 14).
                    // Vertical tremors (Y-axis) are the main interest, but tremors on other axes can be tracked if needed.
                    if (segID == 10 || segID == 14)
                    {
                        if (bone != null)
                        {
                            float currentPositionX = ((Vector3)bone).x;
                            float currentPositionY = ((Vector3)bone).y;
                            float currentPositionZ = ((Vector3)bone).z;

                            // Track Y-axis position for vertical tremors.
                            if (!positionList.ContainsKey("Position " + propertyName + " y")){
                                positionList.Add("Position " + propertyName + " y", new List<float> { currentPositionY });
                            }
                            else{
                                positionList["Position " + propertyName + " y"].Add(currentPositionY);
                            }
                            /*if (!positionList.ContainsKey("Position " + propertyName + " z")){
                                positionList.Add("Position " + propertyName + " z", new List<float> { currentPositionZ });
                            }
                            else {
                                positionList["Position " + propertyName + " z"].Add(currentPositionZ);
                            }
                            /*if (!positionList.ContainsKey("Position " + propertyName + " x")){
                                positionList.Add("Position " + propertyName + " x", new List<float> { currentPositionX });
                            }
                            else {
                                positionList["Position " + propertyName + " x"].Add(currentPositionX);
                            }*/

                            if (!maxMinValues.ContainsKey("Position " + propertyName + " y")){
                                maxMinValues.Add("Position " + propertyName + " y", new Tuple<float?, float?>(currentPositionY, currentPositionY));
                            }
                            /*if (!maxMinValues.ContainsKey("Position " + propertyName + " z")){
                                maxMinValues.Add("Position " + propertyName + " z", new Tuple<float?, float?>(currentPositionZ, currentPositionZ));
                            }
                            if (!maxMinValues.ContainsKey("Position " + propertyName + " x")){
                                maxMinValues.Add("Position " + propertyName + " x", new Tuple<float?, float?>(currentPositionX, currentPositionX));
                            }*/
                        }
                        if (velocity != null)
                        {
                            // Track X-axis velocity for hand movement (front-to-back).
                            float currentVelocityX = ((Vector3)velocity).x;
                            if (!maxMinValues.ContainsKey("Velocity " + propertyName + " x"))
                                maxMinValues.Add("Velocity " + propertyName + " x", new Tuple<float?, float?>(currentVelocityX, currentVelocityX));
                            else
                            {
                                float prevVelocity1 = (float)maxMinValues["Velocity " + propertyName + " x"].Item1;
                                float prevVelocity2 = (float)maxMinValues["Velocity " + propertyName + " x"].Item2;

                                // Update X-axis velocity min and max.
                                if (currentVelocityX > prevVelocity2)
                                    maxMinValues["Velocity " + propertyName + " x"] = new Tuple<float?, float?>(prevVelocity1, currentVelocityX);
                                if (currentVelocityX < prevVelocity1)
                                    maxMinValues["Velocity " + propertyName + " x"] = new Tuple<float?, float?>(currentVelocityX, prevVelocity2);
                            }
                        }
                    }
                }
                else
                {
                    // Activity 31: Hand rest evaluation for right hand (ID 10) and left hand (ID 14). Here, we assess movement in the X, Y, and Z axes and check if the current values stay within a range determined by the first sample +/- a standard deviation threshold.
                    if (segID == 10 || segID == 14)
                    {
                        if (bone != null)
                        {
                            float currentPositionX = ((Vector3)bone).x;
                            float currentPositionY = ((Vector3)bone).y;
                            float currentPositionZ = ((Vector3)bone).z;
                            string error = "";

                            // Update and check Z-axis values against the first sample with a standard deviation range.
                            if (!maxMinValues.ContainsKey("Rest " + propertyName + " z"))
                                maxMinValues.Add("Rest " + propertyName + " z", new Tuple<float?, float?>(currentPositionZ, null));
                            else
                            {
                                float stdZ = extremeDeviationValues31["Rest " + propertyName + " z"].Item2 * (float)stdDeviationQuantity;
                                float firstValueZ = (float)maxMinValues["Rest " + propertyName + " z"].Item1;
                                float inferiorLimit = firstValueZ - stdZ;
                                float superiorLimit = firstValueZ + stdZ;

                                if (currentPositionZ > superiorLimit || currentPositionZ < inferiorLimit)
                                {
                                    errorAct3Rest = true;
                                    error = (currentPositionZ > superiorLimit && currentPositionZ < inferiorLimit)
                                            ? "(> " + superiorLimit.ToString() + ") && (< " + inferiorLimit.ToString() + ")"
                                            : (currentPositionZ > superiorLimit)
                                            ? "(> " + superiorLimit.ToString() + ")"
                                            : "(< " + inferiorLimit.ToString() + ")";
                                    Debug.Log("DIREITA(+): " + currentPositionZ.ToString() + " " + error);
                                }
                            }

                            // Update and check X-axis values against the first sample with a standard deviation range.
                            if (!maxMinValues.ContainsKey("Rest " + propertyName + " x"))
                                maxMinValues.Add("Rest " + propertyName + " x", new Tuple<float?, float?>(currentPositionX, null));
                            else
                            {
                                float stdX = extremeDeviationValues31["Rest " + propertyName + " x"].Item2 * (float)stdDeviationQuantity;
                                float firstValueX = (float)maxMinValues["Rest " + propertyName + " x"].Item1;
                                float inferiorLimit = firstValueX - stdX;
                                float superiorLimit = firstValueX + stdX;

                                if (currentPositionX > superiorLimit || currentPositionX < inferiorLimit)
                                {
                                    errorAct3Rest = true;
                                    error = (currentPositionX > superiorLimit && currentPositionX < inferiorLimit)
                                            ? "(> " + superiorLimit.ToString() + ") && (< " + inferiorLimit.ToString() + ")"
                                            : (currentPositionX > superiorLimit)
                                            ? "(> " + superiorLimit.ToString() + ")"
                                            : "(< " + inferiorLimit.ToString() + ")";
                                    Debug.Log("FRENTE(-): " + currentPositionX.ToString() + " " + error);
                                }
                            }

                            // Update and check Y-axis values against the first sample with a standard deviation range.
                            if (!maxMinValues.ContainsKey("Rest " + propertyName + " y"))
                                maxMinValues.Add("Rest " + propertyName + " y", new Tuple<float?, float?>(currentPositionY, null));
                            else
                            {
                                float stdY = extremeDeviationValues31["Rest " + propertyName + " y"].Item2 * (float)stdDeviationQuantity;
                                float firstValueY = (float)maxMinValues["Rest " + propertyName + " y"].Item1;
                                float inferiorLimit = firstValueY - stdY;
                                float superiorLimit = firstValueY + stdY;

                                if (currentPositionY > superiorLimit || currentPositionY < inferiorLimit)
                                {
                                    errorAct3Rest = true;
                                    error = (currentPositionY > superiorLimit && currentPositionY < inferiorLimit)
                                            ? "(> " + superiorLimit.ToString() + ") && (< " + inferiorLimit.ToString() + ")"
                                            : (currentPositionY > superiorLimit)
                                            ? "(> " + superiorLimit.ToString() + ")"
                                            : "(< " + inferiorLimit.ToString() + ")";
                                    Debug.Log("CIMA(+): " + currentPositionY.ToString() + " " + error);
                                }
                            }
                        }
                    }
                }
            }

            /* Provides real-time feedback based on participant performance in a specific activity.The function analyzes position and velocity data being received in real time, compares it against predefined thresholds, and triggers visual or auditory feedback when performance deviations are detected. */
            private void ProvideFeedback()
            {
                // Capture the current time to calculate feedback intervals.
                currentTime = DateTime.Now;
                
                // Determine threshold time and deviation values based on the current activity. For ACT1 and ACT31, feedback is provided every 5 seconds, for ACT32 every 12 seconds, and for ACT2 every 10 seconds.
                if (activityNumber == 1)
                {
                    thresholdValues = extremeDeviationValues1;
                    threshold_time = 5.0f;
                }
                else if (activityNumber == 31){
                    threshold_time = 5.0f;
                }
                else if (activityNumber == 32)
                {
                    thresholdValues = extremeDeviationValues32;
                    threshold_time = 12.0f;
                }
                else{
                    threshold_time = 10.0f;
                }

                stdDeviationQuantity = activityNumber switch
                {                              // will be scaled as:
                    1 => participantHeight,    // std * height
                    2 => participantHeight,    // std * height
                    31 => 0.165,               // std * 0.165
                    _ => 1.0                   // std
                };

                float minValue;
                float maxValue;
                float inferiorMean;
                float inferiorStdDev;
                float superiorMean;
                float superiorStdDev;
                float inferiorLimit;
                float superiorLimit;
                string error = "";

                // Reference the "Smile" object (emoji) in the scene, used for visual feedback.
                targetObject = GameObject.Find("Smile");

                // Check if the time threshold for providing feedback has been reached.
                if ((currentTime - lastTime).TotalSeconds >= threshold_time)
                {
                    lastTime = DateTime.Now;

                    // Loop through each data entry in maxMinValues to analyze participant performance.
                    foreach (var data in maxMinValues)
                    {
                        if ((activityNumber == 1 || activityNumber == 32) && data.Key.Contains("Position"))
                        {
                            float meanMax;
                            float meanMin;
                            float stdDev;
                            float stdDevMax;
                            float stdDevMin;
                            float maxInferiorLimit;
                            float maxSuperiorLimit;
                            float minInferiorLimit;
                            float minSuperiorLimit;

                            // ACT1 position feedback logic.
                            if (activityNumber == 1)
                            {
                                maxValue = (float)data.Value.Item1;
                                meanMax = (float)thresholdValues[data.Key].Item1.Item1;
                                stdDevMax = (float)thresholdValues[data.Key].Item1.Item2 * (float)stdDeviationQuantity;
                                maxInferiorLimit = (float)participantHeight * (meanMax - stdDevMax);

                                // Check if the max position value is outside the expected range (not as high as it should have been lifted).
                                if (maxValue < maxInferiorLimit)
                                {
                                    // If the error occured on the leg performing, then auditory feedback will be heard. Because in this activity, if the right leg is performing, the left leg is unmoved, so feedback has to refer to the leg moving.
                                    if ((data.Key.Contains("Right") && isRightPerformance) || (data.Key.Contains("Left") && !isRightPerformance))
                                    {
                                        errorAct1Pos = true;
                                        audioClip = (AudioClip)Resources.Load("LevantePe", typeof(AudioClip));
                                        EnqueueAudioClip(audioClip);
                                        errorAct3Rest = true;
                                        error = "(< " + maxInferiorLimit.ToString() + ")";
                                        Debug.Log("POS: " + maxValue.ToString() + " " + error);
                                    }
                                }
                            }
                            // ACT32 position feedback logic.
                            if (activityNumber == 32)
                            {
                                stdDev = (float)thresholdValues[data.Key].Item2.Item2 * (float)stdDeviationQuantity;

                                var temp = positionList[data.Key].ToArray();

                                // Apply the second-order high-pass filter to every position value recorded during the threshold.
                                var filteredValues = ApplyHighPassFilterSecondOrder(positionList[data.Key].ToArray());

                                float peakHighMean = filteredValues.Max();
                                float peakLowMean = filteredValues.Min();

                                // Check for max-min position differences that exceed the std.
                                if (peakHighMean - peakLowMean > stdDev)
                                {
                                    // If the error occured on the leg performing, then auditory feedback will be heard.
                                    if ((data.Key.Contains("Right") && isRightPerformance) || (data.Key.Contains("Left") && !isRightPerformance))
                                    {
                                        Debug.Log("POS DIF: " + (peakHighMean - peakLowMean).ToString() + " > " + stdDev.ToString());
                                        errorAct3Pos = true;
                                    }
                                }
                            }
                        }
                        // ACT2 performance analysis.
                        if (activityNumber == 2)
                        {
                            minValue = (float)data.Value.Item1;
                            maxValue = (float)data.Value.Item2;
                            inferiorMean = (float)extremeDeviationValues2[data.Key].Item1.Item1;
                            inferiorStdDev = (float)extremeDeviationValues2[data.Key].Item1.Item2 * (float)stdDeviationQuantity;
                            superiorMean = (float)extremeDeviationValues2[data.Key].Item2.Item1;
                            superiorStdDev = (float)extremeDeviationValues2[data.Key].Item2.Item2 * (float)stdDeviationQuantity;
                            inferiorLimit = (float)participantHeight * (inferiorMean - inferiorStdDev);
                            superiorLimit = (float)participantHeight * (superiorMean + superiorStdDev);
                            float feetWidth = 0.30f; // empirical value for participant's feet width

                            // Check if the participant's position exceeds the acceptable limits in the Z axis (mediolateral).
                            if (data.Key.Contains("COM z"))
                            {
                                if (minValue < inferiorLimit || maxValue > superiorLimit)
                                {
                                    errorAct2 = true;
                                    audioClip = (AudioClip)Resources.Load("NaoSeInclineLados", typeof(AudioClip));
                                    EnqueueAudioClip(audioClip);
                                    error = (minValue < inferiorLimit && maxValue > superiorLimit)
                                            ? maxValue.ToString() + " (> " +(superiorLimit).ToString() + ") && \n    " + minValue.ToString() + " (< " + (inferiorLimit).ToString() + ")"
                                            : (maxValue > superiorLimit)
                                            ? "DIREITA(+): " + maxValue.ToString() + " (> " + (superiorLimit).ToString() + ")"
                                            : "ESQUERDA(-): " + minValue.ToString() + " (< " + (inferiorLimit).ToString() + ")";
                                    Debug.Log(error);
                                }
                            }
                            // Check if the participant's position exceeds the acceptable limits in the X axis (anteroposterior).
                            if (data.Key.Contains("COM x"))
                            {
                                if (minValue + 0.2 < 1.05 * inferiorLimit || maxValue - 0.2 > 1.05 * superiorLimit)
                                {
                                    errorAct2 = true;
                                    audioClip = (AudioClip)Resources.Load("NaoSeInclineFrente", typeof(AudioClip));
                                    EnqueueAudioClip(audioClip);
                                    error = (minValue < inferiorLimit && maxValue > superiorLimit)
                                            ? maxValue.ToString() + " (> " + superiorLimit.ToString() + ") && \n    " + minValue.ToString() + " (< " + inferiorLimit.ToString() + ")"
                                            : (maxValue > superiorLimit)
                                            ? maxValue.ToString() + " (> " + superiorLimit.ToString() + ")"
                                            : minValue.ToString() + " (< " + inferiorLimit.ToString() + ")";
                                    Debug.Log("FRENTE(-): " + error);
                                }
                            }
                        }

                        // Velocity analysis for ACT1 and ACT32.
                        if (data.Key.Contains("Velocity") && ((activityNumber == 32 && !errorAct3Pos) || (activityNumber == 1 && !errorAct1Pos)))
                        {
                            minValue = (float)data.Value.Item1;
                            maxValue = (float)data.Value.Item2;
                            inferiorMean = (float)thresholdValues[data.Key].Item1.Item1;
                            inferiorStdDev = (float)thresholdValues[data.Key].Item1.Item2 * (float)stdDeviationQuantity;
                            superiorMean = (float)thresholdValues[data.Key].Item2.Item1;
                            superiorStdDev = (float)thresholdValues[data.Key].Item2.Item2 * (float)stdDeviationQuantity;

                            // ACT1 velocity feedback logic.
                            if (activityNumber == 1)
                            {
                                inferiorLimit = (float)participantHeight * (inferiorMean + inferiorStdDev);
                                superiorLimit = (float)participantHeight * (superiorMean - superiorStdDev);

                                // Check if velocity values are outside acceptable limits.
                                if (maxValue < superiorLimit || minValue > inferiorLimit)
                                {
                                    // If the error occured on the leg performing, then auditory feedback will be heard.
                                    if ((data.Key.Contains("Right") && isRightPerformance) || (data.Key.Contains("Left") && !isRightPerformance))
                                    {
                                        errorAct1Vel = true;
                                        audioClip = (AudioClip)Resources.Load("AcelereMovimento", typeof(AudioClip));
                                        EnqueueAudioClip(audioClip);
                                        error = (minValue > inferiorLimit && maxValue < superiorLimit)
                                                ? maxValue.ToString() + " (< " + superiorLimit.ToString() + ") && \n    " + minValue.ToString() + " (> " + inferiorLimit.ToString() + ")"
                                                : (maxValue > superiorLimit)
                                                ? maxValue.ToString() + " (< " + superiorLimit.ToString() + ")"
                                                : minValue.ToString() + " (> " + inferiorLimit.ToString() + ")";
                                        Debug.Log("VEL: " + error);
                                    }
                                }
                            }
                            // ACT32 velocity feedback logic.
                            else if (activityNumber == 32)
                            {
                                inferiorLimit = (float)participantHeight * (inferiorMean - inferiorStdDev);
                                superiorLimit = (float)participantHeight * (superiorMean + superiorStdDev);

                                // Check if velocity values are outside acceptable limits.
                                if (minValue < inferiorLimit || maxValue > superiorLimit)
                                {
                                    // If the error occured on the leg performing, then auditory feedback will be heard.
                                    if ((data.Key.Contains("Right") && isRightPerformance) || (data.Key.Contains("Left") && !isRightPerformance))
                                    {
                                        errorAct3Vel = true;
                                        audioClip = (AudioClip)Resources.Load("DesacelereMovimento", typeof(AudioClip));
                                        EnqueueAudioClip(audioClip);
                                        error = (minValue < inferiorLimit && maxValue > superiorLimit)
                                                ? maxValue.ToString() + " (> " + superiorLimit.ToString() + ") && \n    " + minValue.ToString() + " (< " + inferiorLimit.ToString() + ")"
                                                : (maxValue > superiorLimit)
                                                ? maxValue.ToString() + " (> " + superiorLimit.ToString() + ")"
                                                : minValue.ToString() + " (< " + inferiorLimit.ToString() + ")";
                                        Debug.Log("VEL: " + error);
                                    }
                                }
                            }
                        }
                    }
                    // In case no error was detected and the performance was well executed, then visual feedback will be provided.
                    if ((activityNumber == 1 && !errorAct1Pos && !errorAct1Vel) || (activityNumber == 2 && !errorAct2) || (activityNumber == 32 && !errorAct3Vel && !errorAct3Pos) || (activityNumber == 31 && !errorAct3Rest))
                    {
                        Debug.Log("EVERYTHING OK!");
                        if (targetObject != null)
                        {
                            targetSpriteRenderer = targetObject.GetComponent<SpriteRenderer>();
                            if (targetSpriteRenderer == null)
                                targetSpriteRenderer = targetObject.AddComponent<SpriteRenderer>();
                            targetSpriteRenderer.enabled = true;
                        }
                    }
                    // Otherwise, disable visual feedback if there are errors during ACT1, ACT2 and ACT32 (velocity), in which the auditory feedback has already been provided.
                    else if ((activityNumber == 1 && (errorAct1Pos || errorAct1Vel)) || (activityNumber == 2 && errorAct2) || (activityNumber == 32 && errorAct3Vel))
                    {
                        if (targetObject != null)
                        {
                            targetSpriteRenderer = targetObject.GetComponent<SpriteRenderer>();
                            if (targetSpriteRenderer == null)
                                targetSpriteRenderer = targetObject.AddComponent<SpriteRenderer>();
                            targetSpriteRenderer.enabled = false;
                        }
                    }
                    // As for ACT31 and ACT32 (position), visual feedback is also disabled if there are errors in execution, but auditory feedback is only provided now.
                    else if ((activityNumber == 32 && errorAct3Pos) || (activityNumber == 31 && errorAct3Rest))
                    {
                        audioClip = (AudioClip)Resources.Load("NaoTremaMao", typeof(AudioClip));
                        EnqueueAudioClip(audioClip);
                        if (targetObject != null)
                        {
                            targetSpriteRenderer = targetObject.GetComponent<SpriteRenderer>();
                            if (targetSpriteRenderer == null)
                                targetSpriteRenderer = targetObject.AddComponent<SpriteRenderer>();
                            targetSpriteRenderer.enabled = false;
                        }
                    }

                    errorAct3Rest = false;
                    errorAct1Pos = false;
                    errorAct1Vel = false;
                    errorAct2 = false;
                    errorAct3Pos = false;
                    errorAct3Vel = false;
                    audioClip = null;
                    maxMinValues = new Dictionary<string, Tuple<float?, float?>>();
                    positionList = new Dictionary<string, List<float>>();
                }
            }

            /* Applies a second-order high-pass filter to the given signal, removing low-frequency components and allowing only high-frequency components of the signal to pass through.

            <param name="signal">An array representing the input signal to be filtered.</param>

            <returns>An array representing the filtered signal, with low-frequency components attenuated.</returns> */
            private float[] ApplyHighPassFilterSecondOrder(float[] signal)
            {
                double Q = 2.0;
                double omega = 2 * 3.1416 * cutoffFrequency / samplingFrequency;
                double alpha = Math.Round(Math.Sin(omega), 5) / (2 * Q);

                double a0 = 1 + alpha;
                double a1 = -2 * Math.Round(Math.Cos(omega), 5);
                double a2 = 1 - alpha;
                double b0 = (1.0 + Math.Round(Math.Cos(omega), 5)) / 2.0;
                double b1 = -(1 + Math.Round(Math.Cos(omega), 5));
                double b2 = (1.0 + Math.Round(Math.Cos(omega), 5)) / 2.0;

                float[] filtered = new float[signal.Length];
                filtered[0] = 0;
                filtered[1] = 0;

                for (int i = 2; i < signal.Length; i++)
                {
                    filtered[i] = (float)((b0 / a0) * signal[i] + (b1 / a0) * signal[i - 1] + (b2 / a0) * signal[i - 2]
                                - (a1 / a0) * filtered[i - 1] - (a2 / a0) * filtered[i - 2]);
                }

                return filtered;
            }

            /* Adds an audio clip to the audio queue and starts playing the queue if audio is not already playing.
            
            <param name="clip">The audio clip to be added to the queue for playback.</param> */
            private void EnqueueAudioClip(AudioClip clip)
            {
                audioQueue.Enqueue(clip);
                if (!isPlayingAudio)
                    StartCoroutine(PlayAudioQueue());
            }

            /* Coroutine that sequentially plays all audio clips in the audio queue.

            <returns>An IEnumerator that yields control while audio clips are being played from the queue.</returns> */
            private IEnumerator PlayAudioQueue()
            {
                while (audioQueue.Count > 0)
                {
                    isPlayingAudio = true;
                    AudioClip clip = audioQueue.Dequeue();
                    audioSource.clip = clip;
                    audioSource.Play();
                    yield return new WaitWhile(() => audioSource.isPlaying);
                }
                isPlayingAudio = false;
            }
        // [START] Our code

        bool InitClient()
        {
            var client = Client;

            if (client != null)
            {
                client.FrameDataReceivedAsync -= OnFrameDataReceivedAsync;
                client.FrameDataReceivedAsync += OnFrameDataReceivedAsync;
                client.Disconnected -= OnClientDisconnected;
                client.Disconnected += OnClientDisconnected;

                return true;
            }

            return false;
        }

        protected override void UpdateDevice()
        {
            var animator = Animator;

            if (animator == null)
                return;

            if (animator.avatar != null)
                m_AvatarCache = animator.avatar;

            if (!m_ModelInitialized)
                m_ModelInitialized = InitModel();

            if (!m_ClientInitialized)
                m_ClientInitialized = InitClient();

            if (m_ModelInitialized && m_ClientInitialized && !SyncBuffer.IsSynchronized && IsReady())
            {
                var client = Client;

                client.FrameRate = TakeRecorder.FrameRate;

                var frame = client.GetFrame(m_CharacterID);

                if (frame.SegmentCount == 0)
                    frame = m_PreviousFrame;

                if (frame.SegmentCount != 0)
                    AddFrame(frame, new FrameTimeWithRate(frame.FrameRate, frame.TC.ToFrameTime(frame.FrameRate)));
            }
        }

        bool OnFrameDataReceivedAsync(int characterID, FrameData frame)
        {
            if (SyncBuffer.IsSynchronized && characterID == m_CharacterID)
            {
                if (frame.TC < m_PreviousFrame.TC)
                    ResetSyncBuffer();

                AddFrame(frame, new FrameTimeWithRate(frame.FrameRate, frame.TC.ToFrameTime(frame.FrameRate)));
                return true;
            }

            return false;
        }

        void OnClientDisconnected()
        {
            RestoreTPose();

            m_ClientInitialized = false;

            thresholdValues = new Dictionary<string, Tuple<Tuple<float?, float?>, Tuple<float?, float?>>>();
            maxMinValues = new Dictionary<string, Tuple<float?, float?>>();
            positionList = new Dictionary<string, List<float>>();
            samplingFrequency = 58.8235;
            cutoffFrequency = 3.0;
            errorAct3Rest = false;
            errorAct1Pos = false;
            errorAct1Vel = false;
            errorAct2 = false;
            errorAct3Pos = false;
            errorAct3Vel = false;
            audioClip = null;
            lastTime = DateTime.Now;
        }

        protected override void OnRecordingChanged()
        {
            m_FirstFrameTime = null;

            if (IsRecording)
            {
                var frameRate = TakeRecorder.FrameRate;

                for (int i = 0; i < m_Props.Length; i++)
                {
                    var recorder = m_PropRecorders[i];
                    recorder.Prepare(m_Props[i].gameObject.GetOrAddComponent<Animator>(), frameRate);
                }
            }
        }

        protected override void LiveUpdate()
        {
            base.LiveUpdate();

            for (int i = 0; i < m_Props.Length; ++i)
                m_PropRecorders[i].ApplyFrame(m_Props[i].gameObject.GetOrAddComponent<Animator>());
        }


        /* Processes a given frame of motion capture data, updating the model's body segment positions, rotations, and other key parameters, and providing feedback based on certain activities and human bone segments.

        <param name="frame">The current frame data to process, which contains positions, orientations, velocities, and other information.</param> */
        protected override void ProcessFrame(FrameData frame)
        {
            // Dictionary mapping indexes to body segments.
            Dictionary<int, string> HumanBonesMap = new Dictionary<int, string>
            {
                {0, "Pelvis"},
                {1, "L5"},
                {2, "L3"},
                {3, "T12"},
                {4, "T8"},
                {5, "Neck"},
                {6, "Head"},
                {7, "RightShoulder"},
                {8, "RightUpperArm"},
                {9, "RightForearm"},
                {10, "RightHand"},
                {11, "LeftShoulder"},
                {12, "LeftUpperArm"},
                {13, "LeftForearm"},
                {14, "LeftHand"},
                {15, "RightUpperLeg"},
                {16, "RightLowerLeg"},
                {17, "RightFoot"},
                {19, "LeftUpperLeg"},
                {20, "LeftLowerLeg"},
                {21, "LeftFoot"}
            };

            // [START] Our code
            if (!start)
            {
                // In case the "start" checkbox is (or has been) unchecked, being this checkbox a flag that indicates feedback will no longer be provided.
                errorAct3Rest = false;
                errorAct1Pos = false;
                errorAct1Vel = false;
                errorAct2 = false;
                errorAct3Pos = false;
                errorAct3Vel = false;
                audioClip = null;
                audioSource.clip = null;
                lastTime = DateTime.Now;
                maxMinValues = new Dictionary<string, Tuple<float?, float?>>();
                positionList = new Dictionary<string, List<float>>();
                samplingFrequency = 58.8235;
                cutoffFrequency = 3.0;
                targetObject = GameObject.Find("Smile");
                if (targetObject != null)
                {
                    targetSpriteRenderer = targetObject.GetComponent<SpriteRenderer>();
                    if (targetSpriteRenderer == null)
                        targetSpriteRenderer = targetObject.AddComponent<SpriteRenderer>();
                    targetSpriteRenderer.enabled = true;
                }
            }
            // [END] Our code

            currentTime = DateTime.Now;
            var animator = Animator;

            if (animator == null)
                return;

            var inverseActor = Quaternion.Inverse(animator.transform.rotation);

            try
            {
                var flags = m_Channels;

                // Validate prop tpose and check if prop(s) changed; must be done before caching rotations
                for (int i = 0; i < m_Props.Length; ++i)
                {
                    var prop = m_Props[i].gameObject ? m_Props[i].gameObject.transform : null;

                    TPose tpose = null;

                    // Make sure prop has a TPose component
                    if (prop != null && !prop.TryGetComponent(out tpose))
                    {
                        tpose = prop.gameObject.AddComponent<TPose>();
                        tpose.SaveTPose();
                    }

                    var propID = XsensConstants.MvnBodySegmentCount + i;
                    var model = m_Model[propID].transform;

                    if (model != prop)
                    {
                        if (model != null)
                        {
                            if (model.TryGetComponent<TPose>(out var modelPose))
                                modelPose.RefreshTPose();
                        }

                        if (prop != null)
                            m_Model[propID] = (prop, tpose.Position, tpose.Rotation);
                        else
                            m_Model[propID] = (null, Vector3.zero, Quaternion.identity);
                    }
                }

                // cache original rotations
                for (int i = 0; i < m_Model.Length; ++i)
                {
                    var bone = m_Model[i].transform;
                    m_OriginalRotations[i] = (bone, bone ? bone.localRotation : Quaternion.identity);
                }

                // body segments
                for (int i = 0; i < m_BodySegmentOrder.Length; i++)
                {
                    var bodyID = m_BodySegmentOrder[i];
                    var bone = m_Model[bodyID].transform;

                    if (bone == null)
                        continue;

                    if (frame.Positions != null && frame.Positions.Length != 0)
                    {
                        if (frame.Positions[bodyID] != null)
                        {
                            Vector3? localPosition = null;

                            if (flags.HasFlag(JointFlags.Position))
                            {
                                if (XsBodyAnimationSegment.Pelvis == (XsBodyAnimationSegment)bodyID)
                                {
                                    var invParent = (bone.parent && bone.parent != animator.transform) ?
                                                    Quaternion.Inverse(Quaternion.Inverse(animator.transform.rotation) * bone.parent.transform.rotation) :
                                                    Quaternion.identity;

                                    var newPosition = invParent * (frame.Positions[bodyID] / bone.lossyScale.y);

                                    localPosition = animator.applyRootMotion ? newPosition :
                                        new Vector3(bone.localPosition.x, newPosition.y, bone.localPosition.z);
                                }
                            }

                            if (frame.Orientations[bodyID] != null)
                            {
                                Quaternion? localRotation = null;

                                if (flags.HasFlag(JointFlags.Rotation))
                                {
                                    var parentRotation = bone.parent ? bone.parent.rotation : Quaternion.identity;
                                    var inverseParent = Quaternion.Inverse(inverseActor * parentRotation);
                                    localRotation = inverseParent * (frame.Orientations[bodyID] * m_Model[i].tPoseRotation);

                                    bone.localRotation = localRotation.Value;
                                }

                                Present(bone, localPosition, localRotation, null);
                            }
                        }
                    }

                    // [START] Our code
                    if (HumanBonesMap.ContainsKey(i) && start)
                    {
                        // For ACT1, ACT31 and ACT32, when receiving packets with position and quaternion data.
                        if (activityNumber != 2 && frame.Positions != null && frame.Positions.Length != 0)
                            UpdateMinMaxValues(i, HumanBonesMap, frame.Positions[bodyID], null, null);

                        // For ACT1 or ACT32, when receiving packets with velocity data.
                        if ((activityNumber == 1 || activityNumber == 32) && frame.Velocities != null && frame.Velocities.Length != 0)
                            UpdateMinMaxValues(i, HumanBonesMap, null, frame.Velocities[bodyID], null);
                        
                        // For ACT2, when receiving packets with COM data.
                        if (activityNumber == 2 && frame.PositionsCOM != null && frame.PositionsCOM.Length != 0)
                        UpdateMinMaxValues(i, HumanBonesMap, null, null, frame.PositionsCOM[0]);
                        
                    }

                    // In case "start" checkbox has been checked, start providing feedback.
                    if (start){
                        ProvideFeedback();
                    }
                    // [END] Our code
                }

                // finger segments
                if (frame.SegmentCount > XsensConstants.MvnPropSegmentCount + XsensConstants.MvnBodySegmentCount)
                {
                    for (int i = 0; i < m_FingerSegmentOrder.Length; i++)
                    {
                        var boneID = i + XsensConstants.MvnBodySegmentCount + XsensConstants.MvnPropSegmentCount;
                        var bone = m_Model[boneID].transform;

                        if (bone == null)
                            continue;

                        if (frame.Orientations != null && frame.Orientations.Length != 0)
                        {
                            if (frame.Orientations[boneID] != null)
                            {
                                Quaternion? localRotation = null;

                                if (flags.HasFlag(JointFlags.Rotation))
                                {
                                    var parentRotation = bone.parent ? bone.parent.rotation : Quaternion.identity;
                                    var inverseParent = Quaternion.Inverse(inverseActor * parentRotation);
                                    var newOrientation = frame.Orientations[frame.Orientations.Length - XsensConstants.MvnFingerSegmentCount + i];

                                    localRotation = inverseParent * (newOrientation * m_Model[boneID].tPoseRotation);
                                    bone.localRotation = localRotation.Value;
                                }

                                Present(bone, null, localRotation, null);
                            }
                        }
                    }
                }

                // props
                for (int i = 0; i < m_Props.Length; ++i)
                {
                    var propID = XsensConstants.MvnBodySegmentCount + i;

                    if (i >= frame.NumProps)
                        continue;

                    if (propID > frame.SegmentCount)
                        continue;

                    var prop = m_Props[i].gameObject ? m_Props[i].gameObject.transform : null;

                    if (prop != null)
                    {
                        var type = m_Props[i].segmentType;
                        var parent = GetSegmentTransform(type);

                        if (parent == null)
                            continue;

                        var parentRotation = inverseActor * parent.rotation;

                        var oldPos = prop.position;
                        var oldRot = prop.rotation;

                        if (frame.Orientations[propID] != null && frame.Positions[propID] != null)
                        {
                            var rot = Quaternion.Inverse(parentRotation) * frame.Orientations[propID] * m_Model[propID].tPoseRotation;

                            var propOffset = (frame.Positions[propID] - frame.Positions[(int)type]) * animator.transform.localScale.x;
                            var handRot = Quaternion.Inverse(parentRotation) * frame.Orientations[(int)type] * m_Model[(int)type].tPoseRotation;
                            propOffset = animator.transform.rotation * handRot * propOffset;
                            var pos = parent.position + propOffset;

                            prop.position = pos;
                            prop.rotation = parent.rotation * rot;

                            var recorder = m_PropRecorders[i];

                            recorder.Present(prop.localPosition, prop.localRotation);

                            prop.transform.position = oldPos;
                            prop.transform.rotation = oldRot;

                            if (IsRecording)
                            {
                                var time = frame.TC.ToSeconds(frame.FrameRate);
                                m_FirstFrameTime ??= time;

                                recorder.Record(time - m_FirstFrameTime.Value);
                            }
                        }
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogError(e);
            }
            finally
            {
                for (int i = 0; i < m_OriginalRotations.Length; ++i)
                {
                    var bone = m_OriginalRotations[i].transform;
                    if (bone != null)
                        bone.localRotation = m_OriginalRotations[i].rot;
                }
            }

            m_PreviousFrame = frame;
        }

        internal void RestoreTPose()
        {
            var animator = Animator;

            if (animator == null || animator.avatar == null)
                return;

            //body and fingers t-pose can be restored from the avatar skeleton
            var skeletons = animator.avatar.humanDescription.skeleton;

            if (skeletons == null)
                return;

            var tfs = animator.GetComponentsInChildren<Transform>();
            var dir = new Dictionary<string, Transform>(tfs.Count());

            foreach (var tf in tfs)
            {
                if (!dir.ContainsKey(tf.name))
                    dir.Add(tf.name, tf);
            }

            foreach (var skeleton in skeletons)
            {
                if (!dir.TryGetValue(skeleton.name, out var bone))
                    continue;

                bone.localPosition = skeleton.position;
                bone.localRotation = skeleton.rotation;
                bone.localScale = skeleton.scale;
            }

            for (int i = 0; i < XsensConstants.MvnPropSegmentCount; ++i)
            {
                var propID = XsensConstants.MvnBodySegmentCount + i;
                var prop = m_Model[propID].transform;
                if (prop != null)
                {
                    var tpose = prop.GetComponent<TPose>();
                    if (tpose != null)
                        tpose.RefreshTPose();
                }
            }
        }

        public Transform GetSegmentTransform(XsBodyAnimationSegment segmentType)
        {
            var index = (int)segmentType;
            if (index >= 0 && index < m_Model.Length)
                return m_Model[index].transform;
            return null;
        }

        public override void Write(ITakeBuilder takeBuilder)
        {
            if (Animator == null)
                return;

            base.Write(takeBuilder);

            var animatorBinding = new AnimatorTakeBinding();
            animatorBinding.SetName(Animator.name);

            var track = takeBuilder.CreateTrack<AvatarTrack>("Avatar Track", animatorBinding);
            var clip = track.CreateDefaultClip();

            clip.displayName = "Avatar";
            clip.start = takeBuilder.ContextStartTime;

            var tracks = track.timelineAsset.GetRootTracks();
            double duration = 0;

            foreach (var t in tracks)
            {
                if (t == track)
                    continue;

                if (t.duration > duration)
                    duration = t.duration;
            }

            clip.duration = duration;

            var avatar = Animator.avatar;

            if (avatar == null)
                avatar = m_AvatarCache;

            if (avatar != null)
            {
                var director = GetComponentInParent<PlayableDirector>();

                if (director != null)
                {
                    var asset = clip.asset as AvatarPlayableAsset;
                    asset.Avatar.exposedName = UnityEditor.GUID.Generate().ToString();

                    director.SetReferenceValue(asset.Avatar.exposedName, avatar);
                }
            }

            for (int i = 0; i < m_Props.Length; ++i)
            {
                var propAnimator = m_Props[i].gameObject.GetOrAddComponent<Animator>();

                if (propAnimator != null)
                    takeBuilder.CreateAnimationTrack(propAnimator.name, propAnimator, m_PropRecorders[i].Bake(), alignTime: m_FirstFrameTime);
            }
        }
    }
}