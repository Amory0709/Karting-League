using KartGame.KartSystems;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;
using Random = UnityEngine.Random;
using System;
using Unity.MLAgents.Policies;
using static Codice.CM.Common.CmCallContext;
using Codice.CM.Common;

namespace KartGame.AI
{
    /// <summary>
    /// The KartAgent will drive the inputs for the KartController. This is a template, please implement your own agent by completing the TODO tasks,
    /// but don't be limited by this template, you could change anything in this script to get a better result.
    /// Tips: properties of KartAgent could also be set to public so that you can edit their values in Unity Editor. 
    /// !!! But remember to update your final settings to this script. In competition environment we only read the settings from this script.
    /// </summary>
    // TODO should change class name to KartAgent plus your team name abbreviation,e.g. KartAgentWoW
    public class KartAgent : Agent, IInput
    {
        #region Training Modes
        [Tooltip("Are we training the agent or is the agent production ready?")]
        // use Training mode when you train model, use inferencing mode for competition
        // TODO should be set to Inferencing when you submit this script
        public AgentMode Mode = AgentMode.Training;

        [Tooltip("What is the initial checkpoint the agent will go to? This value is only for inferencing. It is set to a random number in training mode")]
        private int InitCheckpointIndex = 0;
        #endregion

        #region Senses
        [Header("Observation Params")]
        [Tooltip("What objects should the raycasts hit and detect?")]
        public LayerMask Mask;

        [Tooltip("Sensors contain ray information to sense out the world, you can have as many sensors as you need.")]
        public Sensor[] Sensors;

        [Header("Checkpoints")]
        [Tooltip("What are the series of checkpoints for the agent to seek and pass through?")]
        public Collider[] Checkpoints;

        [Tooltip("What layer are the checkpoints on? This should be an exclusive layer for the agent to use.")]
        public LayerMask CheckpointMask;

        [Space]
        [Tooltip("Would the agent need a custom transform to be able to raycast and hit the track? If not assigned, then the root transform will be used.")]
        public Transform AgentSensorTransform;
        #endregion

        #region Rewards
        // TODO Define your own reward/penalty items and set their values
        // For example:
        [Header("Rewards Config")]
        public RewardsConfigInfo RewardsConfig = new RewardsConfigInfo();

        [Header("Runtime Rewards")]
        public RuntimeRewardsInfo RuntimeRewards = new RuntimeRewardsInfo();

        #endregion

        #region Training Parameters

        public TrainingConfig TrainnigConfigInfo = new TrainingConfig();

        #endregion Training Parameters

        #region ResetParams
        [Header("Inference Reset Params")]
        [Tooltip("What is the unique mask that the agent should detect when it falls out of the track?")]
        private LayerMask OutOfBoundsMask;

        [Tooltip("What are the layers we want to detect for the track and the ground?")]
        private LayerMask TrackMask;

        [Tooltip("How far should the ray be when cast? For larger karts - this value should be larger too.")]
        private float GroundCastDistance;
        #endregion

        #region Inspectors

        [Header("Action")]
        public int Action0 = -3;

        public int Action1 = -3;

        [Header("Speed")]
        [Tooltip("Kart current Speed")]
        public float Speed = 0;

        [Header("Stucked")]
        [Tooltip("If the kart is stucked")]
        public bool Stucked = false;
        [Tooltip("If the kart is stucked")]
        public float StuckedDistance = 5f;

        [Header("Collided")]
        [Tooltip("If the kart is collided")]
        public bool Collided = false;

        [Tooltip("If the kart is collided")]
        public float DirectionRewards = 0;
        #endregion Inspectors

        #region Private

        private ArcadeKart m_Kart;
        private bool m_Acceleration;
        private bool m_Brake;
        private float m_Steering;
        public int m_CheckpointIndex;
        private int m_targetingCheckpointIndex;

        private bool m_EndEpisode;

        private float m_startTime = 0;
        private int m_loopCount = 0;
        private bool m_startCheck = false;
        private float m_speedTimer = 0;
        private Vector3 m_checkPosition = Vector3.zero;
        private int m_stuckedFrames = 0;
        private int m_collidedFrames = 0;
        private int m_wrongDirectionFrames = 0;
        public float m_wrongDirectionAccumulate = 0;
        public float m_rightDirectionFrames = 0;

        #endregion Private

        [System.Serializable]
        public class TrainingConfig
        {
            [Header("Training Control")]
            [Tooltip("How many loops finished that will end the episode")]
            public int EpisodeMaxLoopCount = 1;

            [Tooltip("How many frames will it be considered as stucked when kart moves less than a distance")]
            public int StuckedThreshold = 50;

            [Tooltip("How many frames will it can be stucked, it will end episode if stucked framres exceeds this MaxStuckedFrames")]
            public int MaxStuckedFrames = 200;

            [Tooltip("How many frames will it can be collided, it will end episode if collided frames exceeds this MaxCollidedFrames")]
            public int MaxCollidedFrames = 1;

            [Tooltip("Max Panelty in wrong direction, it will end episode if wrong Panelty  exceeds this MaxWrongDirectionPanelty")]
            public int MaxWrongDirectionFrames = 20;

            public int MaxRightDirectionFrames = -1;

            public float MaxNegativeRewards = -15000f;


            [Header("Training Parameters")]

            [Tooltip("How much random deviation range is when start training?")]
            public int Deviation_TraningStart = 0;

            [Header("RaycastMaxDistance")]
            [Tooltip("The max detected distance in the sensers directions")]
            public float RaycastMaxDistance = 100f;

            [Header("CollisionRadius")]
            [Tooltip("When sensor rays hit others and the distance is less than this value, we consider they are collided, should be similar to the car's size")]
            public float CollisionRadius = 1.0f;

        }

        [System.Serializable]
        public class RuntimeRewardsInfo
        {


            [Header("Total Penalty")]
            public float TotalTriggerCollisionPenalty = 0;
            public float TotalStayCollisionPenalty = 0;
            public float TotalStuckPenalty = 0;

            [Header("Total Rewards")]
            public float TotalPassCheckpointReward = 0;
            public float TotalTowardsCheckpointReward = 0;
            public float TotalSpeedReward = 0;
            public float TotalAccelerationReward = 0;
            public float TotalSpeedTowardNextCheckpointReward = 0;
            public float TotalLoopSpentTimeReward = 0;
            public float TotalLastAccumulatedReward = 0;
            public float TotalFinishCheckpointAverageSpeedReward = 0f;

            [Range(-1000f, 1000f)]
            public float OverallRewards = 0f;

        }

        [System.Serializable]
        public class RewardsConfigInfo
        {
            [Header("Penalty Config")]

            [Tooltip("What penalty is given when the agent enter collisions?")]
            public float TriggerCollisionPenalty = -0.01f;

            [Tooltip("What penalty is given when the agent stays in collisions?")]
            public float StayCollisionPenalty = -0.002f;

            [Tooltip("What penalty is given when the agent crashes?")]
            public float StuckPenalty = -0.3f;

            [Header("Reward Config")]
            [Tooltip("How much reward is given when the agent successfully passes the checkpoints?")]
            public float PassCheckpointReward = 1f;

            [Tooltip("How much reward is given when the agent successfully passes the checkpoints?")]
            public float FinishCheckpointAverageSpeedReward = 100f;

            [Tooltip("Should typically be a small value, but we reward the agent for moving in the right direction.")]
            public float TowardsCheckpointReward = 0f;

            [Tooltip("Typically if the agent moves faster, we want to reward it for finishing the track quickly.")]
            public float SpeedReward = 0.005f;

            [Tooltip("Reward the agent when it keeps accelerating")]
            public float AccelerationReward = 0.001f;

            [Tooltip("If the agent moves faster towards next checkpoint.")]
            public float SpeedTowardNextCheckpointReward = 0.005f;

            [Tooltip("How much reward is given according to the spent time after the agent successfully finishes a loop?")]
            public float LoopSpentTimeRewardFactor = 5000f;

            [Tooltip("Whenever kart runs(not collided or stucked) it will get this reward")]
            public float LastAccumulatedReward = 0.001f;
        }

        #region Initialization
        private void Awake()
        {
            m_Kart = GetComponent<ArcadeKart>();

            AgentSensorTransform = transform.Find("MLAgent_Sensors");
            SetBehaviorParameters();
            SetDecisionRequester();
        }

        private void SetBehaviorParameters()
        {
            var behaviorParameters = GetComponent<BehaviorParameters>();
            behaviorParameters.UseChildSensors = true;
            behaviorParameters.ObservableAttributeHandling = ObservableAttributeOptions.Ignore;
            // TODO set other behavior parameters according to your own agent implementation, especially following ones:
            //behaviorParameters.BehaviorType = BehaviorType.Default;
            //behaviorParameters.BehaviorName = "First-SG-Driver";
            //behaviorParameters.BrainParameters.VectorObservationSize = 13; // size of the ML input data, should be the same as the `AddObservation` number
            //behaviorParameters.BrainParameters.NumStackedVectorObservations = 4;
            //behaviorParameters.BrainParameters.ActionSpec = ActionSpec.MakeDiscrete(3, 2); // format of the ML model output data [0, 1, 2] [0, 1, 2]

        }
        private void InitializeTotalRewards()
        {
            RuntimeRewards.TotalAccelerationReward = 0;
            RuntimeRewards.TotalLastAccumulatedReward = 0;
            RuntimeRewards.TotalLoopSpentTimeReward = 0;
            RuntimeRewards.TotalPassCheckpointReward = 0;
            RuntimeRewards.TotalSpeedReward = 0;
            RuntimeRewards.TotalSpeedTowardNextCheckpointReward = 0;
            RuntimeRewards.TotalStayCollisionPenalty = 0;
            RuntimeRewards.TotalStuckPenalty = 0;
            RuntimeRewards.TotalTowardsCheckpointReward = 0;
            RuntimeRewards.TotalTriggerCollisionPenalty = 0;
            RuntimeRewards.TotalFinishCheckpointAverageSpeedReward = 0;
            RuntimeRewards.OverallRewards = 0f;
        }
        private void InitializeOtherParameters()
        {
            m_speedTimer = Time.time;
            m_startCheck = false;
            Stucked = false;
            m_loopCount = 0;
            Collided = false;
            Speed = 0;
            m_stuckedFrames = 0;
            m_collidedFrames = 0;
            m_wrongDirectionFrames = 0;
            m_wrongDirectionAccumulate = 0f;
            DirectionRewards = 0f;
            m_rightDirectionFrames = 0;
        }


        private void SetDecisionRequester()
        {
            var decisionRequester = GetComponent<DecisionRequester>();
            //TODO set your decision requester
            decisionRequester.DecisionPeriod = 1;
            decisionRequester.TakeActionsBetweenDecisions = true;
        }

        private void InitialiseResetParameters()
        {
            OutOfBoundsMask = LayerMask.GetMask("Ground");
            TrackMask = LayerMask.GetMask("Track");
            GroundCastDistance = 1f;
        }

        private void InitializeSenses()
        {
            // TODO Define your own sensors
            // make sure you are choosing from our predefined sensors, otherwise it won't work in competition
            // predefined:
            // "MLAgent_Sensors/0" "MLAgent_Sensors/15" "MLAgent_Sensors/30" "MLAgent_Sensors/45" "MLAgent_Sensors/60" "MLAgent_Sensors/75" "MLAgent_Sensors/90"
            // "MLAgent_Sensors/-15" "MLAgent_Sensors/-30" "MLAgent_Sensors/-45" "MLAgent_Sensors/-60" "MLAgent_Sensors/-75" "MLAgent_Sensors/-90"
            Sensors = new Sensor[9];
            Sensors[0] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/-90"),
                HitValidationDistance = 1f,
                RayDistance = 50
            };

            Sensors[1] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/-60"),
                HitValidationDistance = 1.5f,
                RayDistance = 50
            };
            Sensors[2] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/-30"),
                HitValidationDistance = 1.8f,
                RayDistance = 50
            };
            Sensors[3] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/0"),
                HitValidationDistance = 3f,
                RayDistance = 50
            };
            Sensors[4] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/30"),
                HitValidationDistance = 1.8f,
                RayDistance = 50
            };
            Sensors[5] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/60"),
                HitValidationDistance = 1.5f,
                RayDistance = 50,

            };
            Sensors[6] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/90"),
                HitValidationDistance = 1f,
                RayDistance = 100
            };
            Sensors[7] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/45"),
                HitValidationDistance = 1.7f,
                RayDistance = 50
            };
            Sensors[8] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/-45"),
                HitValidationDistance = 1.7f,
                RayDistance = 50
            };

            // set Mask
            Mask = LayerMask.GetMask("Default", "Ground", "Environment", "Track");

            // set Checkpoints
            Checkpoints = GameObject.Find("Checkpoints").transform.GetComponentsInChildren<Collider>();

            // set CheckpointMask
            CheckpointMask = LayerMask.GetMask("CompetitionCheckpoints", "TrainingCheckpoints");
        }


        public override void OnEpisodeBegin()
        {
            m_Steering = 0f;
            InitialiseResetParameters();
            InitializeSenses();
            switch (Mode)
            {
                case AgentMode.Training:
                    Debug.Log($"[EpisodeBegin]");

                    m_CheckpointIndex = Random.Range(0, Checkpoints.Length - 1);
                    InitCheckpointIndex = m_CheckpointIndex;
                    m_targetingCheckpointIndex = (m_CheckpointIndex + 1) % Checkpoints.Length;
                    var collider = Checkpoints[m_CheckpointIndex];

                    transform.localRotation
                        = collider.transform.rotation
                        * Quaternion.Euler(
                            //Random.Range(-TrainigConfigInfo.RandomDeviation_TraningStart, TrainigConfigInfo.RandomDeviation_TraningStart),
                            //Random.Range(-TrainigConfigInfo.RandomDeviation_TraningStart, TrainigConfigInfo.RandomDeviation_TraningStart),
                            0,
                            TrainnigConfigInfo.Deviation_TraningStart,
                            0);
                    transform.position = collider.transform.position;
                    m_checkPosition = transform.position;
                    m_Kart.Rigidbody.velocity = default;
                    m_Acceleration = false;
                    m_Brake = false;
                    m_Steering = 0f;
                    InitializeOtherParameters();
                    InitializeTotalRewards();
                    break;
            }
            m_startTime = Time.time;
        }

        private void Start()
        {
            // If the agent is training, then at the start of the simulation, pick a random checkpoint to train the agent.
            OnEpisodeBegin();

            if (Mode == AgentMode.Inferencing)
                m_CheckpointIndex = InitCheckpointIndex;
        }
        #endregion Initialization

        #region UpdateStatus
        private void Update()
        {
            switch (Mode)
            {
                case AgentMode.Training:
                    //if (m_EndEpisode)
                    //{
                    //    m_EndEpisode = false;
                    //    EndEpisode();
                    //}
                    break;
            }
        }
        private void DrawDetectingRays()
        {
            int nextCheckpointIndex = (m_CheckpointIndex + 1) % Checkpoints.Length;
            Debug.DrawLine(AgentSensorTransform.position, Checkpoints[m_targetingCheckpointIndex].transform.position, Color.green);
            Debug.DrawLine(AgentSensorTransform.position, Checkpoints[nextCheckpointIndex].transform.position, Color.cyan);
            Debug.DrawLine(AgentSensorTransform.position, Checkpoints[m_CheckpointIndex].transform.position, Color.magenta);
            // Add your rewards/penalties
            for (int i = 0; i < Sensors.Length; i++)
            {
                if (i == 0 || i == 1 || i == 5 || i == 6)
                {
                    continue;
                }
                var current = Sensors[i];
                var xform = current.Transform;
                var hit = Physics.Raycast(AgentSensorTransform.position, xform.forward, out var hitInfo,
                    TrainnigConfigInfo.RaycastMaxDistance, Mask, QueryTriggerInteraction.Ignore);

                var endPoint = AgentSensorTransform.position + xform.forward * TrainnigConfigInfo.RaycastMaxDistance;
                bool drawKartRadius = true;
                Color color = Color.blue;
                if (hit)
                {
                    if (hitInfo.distance <= TrainnigConfigInfo.CollisionRadius)
                    {
                        color = Color.red;
                        drawKartRadius = false;
                    }
                    else
                    {
                        endPoint = hitInfo.point;

                    }
                }

                Debug.DrawLine(AgentSensorTransform.position, endPoint, color);
                if (drawKartRadius)
                {
                    Debug.DrawLine(AgentSensorTransform.position, AgentSensorTransform.position + xform.forward * TrainnigConfigInfo.CollisionRadius, Color.yellow);
                }

            }
        }

        private void LateUpdate()
        {
            switch (Mode)
            {
                case AgentMode.Inferencing:
                    // We want to place the agent back on the track if the agent happens to launch itself outside of the track.
                    if (Physics.Raycast(transform.position + Vector3.up, Vector3.down, out var hit, GroundCastDistance, TrackMask)
                        && ((1 << hit.collider.gameObject.layer) & OutOfBoundsMask) > 0)
                    {
                        // Reset the agent back to its last known agent checkpoint
                        var checkpoint = Checkpoints[m_CheckpointIndex].transform;
                        transform.localRotation = checkpoint.rotation;
                        transform.position = checkpoint.position;
                        m_Kart.Rigidbody.velocity = default;
                        m_Steering = 0f;
                        m_Acceleration = m_Brake = false;
                    }
                    break;
            }
        }

        private void FixedUpdate()
        {
            DrawDetectingRays();

            if (Mode == AgentMode.Inferencing)
            {
                return;
            }
            RuntimeRewards.OverallRewards
                   = RuntimeRewards.TotalSpeedReward
                   + RuntimeRewards.TotalTowardsCheckpointReward
                   + RuntimeRewards.TotalPassCheckpointReward
                   + RuntimeRewards.TotalAccelerationReward
                   + RuntimeRewards.TotalLoopSpentTimeReward
                   + RuntimeRewards.TotalLastAccumulatedReward
                   + RuntimeRewards.TotalSpeedTowardNextCheckpointReward
                   + RuntimeRewards.TotalFinishCheckpointAverageSpeedReward
                   + RuntimeRewards.TotalStuckPenalty
                   + RuntimeRewards.TotalStayCollisionPenalty
                   + RuntimeRewards.TotalTriggerCollisionPenalty;
            if (RuntimeRewards.OverallRewards < TrainnigConfigInfo.MaxNegativeRewards)
            {
                Debug.Log("[Quick Stop Training Episode Ended] MaxNegativeRewards Exceeded");
                m_EndEpisode = true;
            }
            if (m_EndEpisode)
            {
                m_EndEpisode = false;

                Debug.Log($"[Episode Ended] [FinalRewards]:{RuntimeRewards.OverallRewards}");
                Debug.Log($"[Episode Ended],[Penalty] " +
                    $"Stuck:{RuntimeRewards.TotalStuckPenalty}, " +
                    $"StayCollision:{RuntimeRewards.TotalStayCollisionPenalty}, " +
                    $"TriggerCollision:{RuntimeRewards.TotalTriggerCollisionPenalty}.");
                Debug.Log($"[Episode Ended], [Reward]" +
                   $"Speed:{RuntimeRewards.TotalSpeedReward}, " +
                   $"TowardsCheckpoint:{RuntimeRewards.TotalTowardsCheckpointReward}, " +
                   $"PassCheckpoint:{RuntimeRewards.TotalPassCheckpointReward}, " +
                   $"Acceleration:{RuntimeRewards.TotalAccelerationReward}, " +
                   $"LoopSpentTime:{RuntimeRewards.TotalLoopSpentTimeReward}, " +
                   $"LastAccumulate:{RuntimeRewards.TotalLastAccumulatedReward},  " +
                   $"SpeedTowardNextCheckpoint:{RuntimeRewards.TotalSpeedTowardNextCheckpointReward} " +
                   $"FinishCheckpointAverageSpeedReward:{RuntimeRewards.TotalFinishCheckpointAverageSpeedReward}");
                EndEpisode();
            }

            UpdateKartStatusRewards();
            Speed = m_Kart.LocalSpeed();
        }

        #region Collisions

        private void OnCollisionEnter(Collision collision)
        {
            float speed = m_Kart.LocalSpeed();
            Debug.Log($"[OnCollisionEnter],kart speed:{speed}, kart foward:{m_Kart.transform.forward}");
            Collided = true;
            for (int i = 0; i < collision.contactCount; i++)
            {
                var contact = collision.GetContact(i);
                var direction = (contact.point - transform.position).normalized;
                float speedFactor = 0;
                if (speed > 0f)
                {
                    speedFactor = 1f;
                }
                else if (speed < 0f)
                {
                    speedFactor = -1f;
                }
                var factor = Vector3.Dot(direction, m_Kart.transform.forward * speedFactor);
                if (factor > 0f)
                {
                    //AddReward(factor * RewardsConfig.TriggerCollisionPenalty, ref RuntimeRewards.TotalTriggerCollisionPenalty);
                }
            }
        }
        private void OnCollisionStay(Collision collision)
        {
            float speed = m_Kart.LocalSpeed();
            Debug.Log($"[OnCollisionStay],kart speed:{speed}, kart foward:{m_Kart.transform.forward}");
            for (int i = 0; i < collision.contactCount; i++)
            {
                var contact = collision.GetContact(i);
                var direction = (contact.point - transform.position).normalized;
                //float speedFactor = 0;
                //if (speed > 0f)
                //{
                //    speedFactor = 1f;
                //}
                //else if (speed < 0f)
                //{
                //    speedFactor = -1f;
                //}
                var factor = Vector3.Dot(direction, m_Kart.transform.forward * speed);

                AddReward(factor * RewardsConfig.StayCollisionPenalty, ref RuntimeRewards.TotalStayCollisionPenalty);
            }
            AddReward(RewardsConfig.StayCollisionPenalty, ref RuntimeRewards.TotalStayCollisionPenalty);
            //AddReward(RewardsConfig.StayCollisionPenalty * m_collidedFrames, ref RuntimeRewards.TotalStayCollisionPenalty);
            m_collidedFrames++;
            if (m_collidedFrames > TrainnigConfigInfo.MaxCollidedFrames)
            {
                Debug.Log("[Quick Stop Training Episode Ended] MaxCollidedFrames Exceeded");
                m_EndEpisode = true;
            }
        }

        private void OnCollisionExit(Collision collision)
        {
            Debug.Log("[OnCollisionExit]");
            m_collidedFrames = 0;
            Collided = false;
        }
        private void OnTriggerExit(Collider other)
        {
            if (!m_startCheck)
            {
                m_startCheck = true;
                return;
            }
            var maskedValue = 1 << other.gameObject.layer;
            var triggered = maskedValue & CheckpointMask;
            if (triggered > 0)
            {
                FindCheckpointIndex(other, out var index);

                Debug.Log($"[Exit Checkpoint]CheckPointIndex:[{index}]");
                var directionDot = Vector3.Dot((transform.forward * m_Kart.LocalSpeed()).normalized, other.transform.forward);
                AddReward(RewardsConfig.PassCheckpointReward * directionDot, ref RuntimeRewards.TotalPassCheckpointReward);


                bool rightDirection = directionDot > 0;
                if (!rightDirection)
                {
                    m_targetingCheckpointIndex = index;
                    AddReward(-RewardsConfig.PassCheckpointReward, ref RuntimeRewards.TotalPassCheckpointReward);
                }
                else
                {
                    m_targetingCheckpointIndex = (index + 1) % Checkpoints.Length;
                    AddReward(RewardsConfig.PassCheckpointReward, ref RuntimeRewards.TotalPassCheckpointReward);
                }
            }
        }

        void OnTriggerEnter(Collider other)
        {
            if (!m_startCheck)
            {
                return;
            }
            var maskedValue = 1 << other.gameObject.layer;
            var triggered = maskedValue & CheckpointMask;
            if (triggered > 0)
            {
                FindCheckpointIndex(other, out var index);
                Debug.Log($"[Enter Checkpoint] index:{index}, m_CheckpointIndex:{m_CheckpointIndex},m_targetingCheckpointIndex:{m_targetingCheckpointIndex}");

                if (index == -1)
                {
                    return;
                }
                // specify targeting CheckpointIndex
                m_targetingCheckpointIndex = (index + 1) % Checkpoints.Length;

                int nextCheckpointIndex = (m_CheckpointIndex + 1) % Checkpoints.Length;

                var directionDot = Vector3.Dot((transform.forward * m_Kart.LocalSpeed()).normalized, other.transform.forward);

                if (directionDot > 0)
                {
                    if (index == nextCheckpointIndex)
                    {
                        // successfuly passed a new checkpoint
                        var sectionSpentTime = Time.time - m_speedTimer;
                        Debug.Log($"[Passed a new Checkpoint] [index]:{index}, [spend time]:{sectionSpentTime}");
                        AddReward(RewardsConfig.PassCheckpointReward, ref RuntimeRewards.TotalPassCheckpointReward);
                        AddReward(RewardsConfig.FinishCheckpointAverageSpeedReward / sectionSpentTime, ref RuntimeRewards.TotalFinishCheckpointAverageSpeedReward);
                        m_speedTimer = Time.time;
                        if (index == InitCheckpointIndex)
                        {
                            // a loop
                            float spentTime = Time.time - m_startTime;
                            Debug.Log($"[Finish loop] Loop number({m_loopCount}), spend time:{spentTime}");
                            m_startTime = Time.time;
                            AddReward(RewardsConfig.LoopSpentTimeRewardFactor / spentTime, ref RuntimeRewards.TotalLoopSpentTimeReward);
                            m_loopCount++;
                            if (m_loopCount >= TrainnigConfigInfo.EpisodeMaxLoopCount)
                            {
                                Debug.Log("[Stop Training Episode Ended] Rich max loop");
                                m_EndEpisode = true;
                            }
                        }
                        m_CheckpointIndex = index;
                    }
                }
                else // go backtoward 
                {
                    // paneltly according to checkpoint index diff
                    int backwardCount = (index - nextCheckpointIndex - Checkpoints.Length) % Checkpoints.Length;
                    AddReward(RewardsConfig.PassCheckpointReward * backwardCount, ref RuntimeRewards.TotalPassCheckpointReward);
                }
            }
        }

        #endregion Collisions

        #endregion UpdateStatus

        #region Reward Strategy
        private void UpdateKartStatusRewards()
        {
            //average of next to checkpoint direction rewards
            var nextCollider = Checkpoints[m_targetingCheckpointIndex];
            var nextNext = Checkpoints[(m_targetingCheckpointIndex + 1) % Checkpoints.Length];
            var direction1 = (nextCollider.transform.position - m_Kart.transform.position).normalized;
            var direction2 = (nextNext.transform.position - m_Kart.transform.position).normalized;
            //var rewardfactor11 = Vector3.Dot(m_Kart.Rigidbody.velocity, direction1);
            var reward = Vector3.Dot(m_Kart.transform.forward * m_Kart.LocalSpeed(), direction1);
            //var reward = rewardfactor11 * RewardsConfig.TowardsCheckpointReward * m_Kart.LocalSpeed();
            DirectionRewards = reward;
            AddReward(reward, ref RuntimeRewards.TotalTowardsCheckpointReward);
            if (!Stucked && !Collided)
            {
                if (reward < 0)
                {
                    m_wrongDirectionFrames++;
                    m_wrongDirectionAccumulate += reward;
                    //AddReward(reward21 * RewardsConfig.SpeedTowardNextCheckpointReward * m_wrongDirectionFrames, ref RuntimeRewards.TotalSpeedTowardNextCheckpointReward);

                }
                else
                {
                    m_rightDirectionFrames++;
                    //AddReward(reward21 * RewardsConfig.SpeedTowardNextCheckpointReward, ref RuntimeRewards.TotalSpeedTowardNextCheckpointReward);
                }
            }
            var reward12 = Vector3.Dot(m_Kart.Rigidbody.velocity.normalized, direction1);
            var reward22 = Vector3.Dot(m_Kart.transform.forward.normalized * m_Kart.LocalSpeed(), direction2);

            // Add rewards if the agent is heading in the right direction
            AddReward((m_Acceleration && !m_Brake ? 1.0f : 0.0f) * RewardsConfig.AccelerationReward, ref RuntimeRewards.TotalAccelerationReward);
            AddReward(m_Kart.LocalSpeed() * RewardsConfig.SpeedReward, ref RuntimeRewards.TotalSpeedReward);

            CheckStucked();
            AddReward(RewardsConfig.LastAccumulatedReward, ref RuntimeRewards.TotalLastAccumulatedReward);


            if (m_wrongDirectionFrames >= TrainnigConfigInfo.MaxWrongDirectionFrames)
            {
                Debug.Log("[Quick Stop Training Episode Ended] MaxWrongDirectionFrames Exceeded");
                m_EndEpisode = true;
            }
            if (m_rightDirectionFrames > TrainnigConfigInfo.MaxRightDirectionFrames && (TrainnigConfigInfo.MaxRightDirectionFrames > 0))
            {
                Debug.Log("[Quick Stop Training Episode Ended] MaxRightDirectionFrames Exceeded");
                m_EndEpisode = true;
            }
        }
        private void CheckStucked()
        {
            float moved = (transform.position - m_checkPosition).magnitude;

            if (moved > StuckedDistance)
            {
                m_checkPosition = transform.position;
                Stucked = false;
                m_stuckedFrames = 0;
            }
            else
            {
                m_stuckedFrames++;
                if (m_stuckedFrames > TrainnigConfigInfo.StuckedThreshold)
                {
                    Stucked = true;
                    AddReward(RewardsConfig.StuckPenalty, ref RuntimeRewards.TotalStuckPenalty);
                }
                if (Stucked && m_stuckedFrames > (TrainnigConfigInfo.MaxStuckedFrames + TrainnigConfigInfo.StuckedThreshold))
                {
                    Debug.Log("[Quick Stop Training Episode Ended] MaxStuckedFrames Exceeded");
                    m_EndEpisode = true;
                }
            }
        }


        private void AddReward(float reward, ref float total)
        {
            AddReward(reward);
            total += reward;
        }
        #endregion Reward Strategy

        public InputData GenerateInput()
        {
            return new InputData
            {
                Accelerate = m_Acceleration,
                Brake = m_Brake,
                TurnInput = m_Steering
            };
        }

        void FindCheckpointIndex(Collider checkPoint, out int index)
        {
            for (int i = 0; i < Checkpoints.Length; i++)
            {
                if (Checkpoints[i].GetInstanceID() == checkPoint.GetInstanceID())
                {
                    index = i;
                    return;
                }
            }
            index = -1;
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            // TODO Add your observations
            // For example
            sensor.AddObservation(m_Kart.LocalSpeed());
            sensor.AddObservation(m_Kart.Rigidbody.velocity);
            var nextCollider = Checkpoints[m_targetingCheckpointIndex];
            var directionNext = (nextCollider.transform.position - m_Kart.transform.position).normalized;
            var directionAngle = Vector3.Angle(directionNext, m_Kart.transform.forward);
            sensor.AddObservation(directionNext);
            sensor.AddObservation(directionAngle);
          
            foreach (var current in Sensors)
            {
                var xform = current.Transform;
                var hit = Physics.Raycast(AgentSensorTransform.position, xform.forward, out var hitInfo,
                    current.RayDistance, Mask, QueryTriggerInteraction.Ignore);
                var angle = Vector3.Angle(xform.forward, m_Kart.Rigidbody.velocity);

                sensor.AddObservation(hit ? hitInfo.distance : current.RayDistance);
                sensor.AddObservation(angle);  
            }
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            base.OnActionReceived(actions);
            InterpretDiscreteActions(actions);


            //Find the next checkpoint when registering the current checkpoint that the agent has passed.
        }

        private void InterpretDiscreteActions(ActionBuffers actions)
        {
            m_Steering = actions.DiscreteActions[0] - 1f;
            m_Acceleration = actions.DiscreteActions[1] >= 1.0f;
            m_Brake = actions.DiscreteActions[1] < 1.0f;
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            base.Heuristic(actionsOut);
            ActionSegment<int> discreteActions = actionsOut.DiscreteActions;
            var steer = Input.GetAxis("Horizontal");
            if (steer == 0)
            {
                discreteActions[0] = 0;
            }
            else
            {
                discreteActions[0] = steer > 0 ? 1 : -1;
            }
            if (Input.GetButton("Accelerate"))
            {
                discreteActions[1] = 1;
            }
            if (Input.GetButton("Brake"))
            {
                discreteActions[1] = 0;
            }
        }

    }
}