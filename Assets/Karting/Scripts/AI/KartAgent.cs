using KartGame.KartSystems;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;
using Random = UnityEngine.Random;
using System;
using Unity.MLAgents.Policies;

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
        public AgentMode Mode = AgentMode.Inferencing;

        [Tooltip("What is the initial checkpoint the agent will go to? This value is only for inferencing. It is set to a random number in training mode")]
        private ushort InitCheckpointIndex = 0;
        #endregion

        #region Senses
        [Header("Observation Params")]
        [Tooltip("What objects should the raycasts hit and detect?")]
        private LayerMask Mask;

        [Tooltip("Sensors contain ray information to sense out the world, you can have as many sensors as you need.")]
        private Sensor[] Sensors;

        [Header("Checkpoints")]
        [Tooltip("What are the series of checkpoints for the agent to seek and pass through?")]
        private Collider[] Checkpoints;

        [Tooltip("What layer are the checkpoints on? This should be an exclusive layer for the agent to use.")]
        private LayerMask CheckpointMask;

        [Space]
        [Tooltip("Would the agent need a custom transform to be able to raycast and hit the track? If not assigned, then the root transform will be used.")]
        private Transform AgentSensorTransform;
        #endregion

        #region Rewards
        // TODO Define your own reward/penalty items and set their values
        // For example:
        [Header("Rewards")]
        [Tooltip("What penalty is given when the agent crashes?")]
        private float Penalty = -1;

        [Tooltip("How much reward is given when the agent successfully passes the checkpoints?")]
        private float PassCheckpointReward = 1f;
        #endregion

        #region ResetParams
        [Header("Inference Reset Params")]
        [Tooltip("What is the unique mask that the agent should detect when it falls out of the track?")]
        private LayerMask OutOfBoundsMask;

        [Tooltip("What are the layers we want to detect for the track and the ground?")]
        private LayerMask TrackMask;

        [Tooltip("How far should the ray be when cast? For larger karts - this value should be larger too.")]
        private float GroundCastDistance;
        #endregion

        private ArcadeKart m_Kart;
        private bool m_Acceleration;
        private bool m_Brake;
        private float m_Steering;
        public int m_LastHitCheckpointIndex;
        private int m_targetingCheckpointIndex;
        private int m_lastLeftCheckpointIndex = -1;

        private bool m_EndEpisode;
        private float m_LastAccumulatedReward;

        private void Awake()
        {
            m_Kart = GetComponent<ArcadeKart>();
            AgentSensorTransform = transform.Find("MLAgent_Sensors");
            SetBehaviorParameters();
            SetDecisionRequester();
        }

        private void SetBehaviorParameters(){
            var behaviorParameters = GetComponent<BehaviorParameters>();
            behaviorParameters.UseChildSensors = true;
            behaviorParameters.ObservableAttributeHandling = ObservableAttributeOptions.Ignore;
            // TODO set other behavior parameters according to your own agent implementation, especially following ones:
            behaviorParameters.BehaviorType = BehaviorType.Default;
            behaviorParameters.BehaviorName = "SixGods";
            behaviorParameters.BrainParameters.VectorObservationSize = 34; // size of the ML input data, should be the same as the `AddObservation` number
            behaviorParameters.BrainParameters.NumStackedVectorObservations = 20;
            behaviorParameters.BrainParameters.ActionSpec = ActionSpec.MakeDiscrete(3, 2); // format of the ML model output data [0, 1, 2] [0, 1, 2]

        }

        private void InitializeTotalRewards()
        {
            RuntimeRewards.TotalAccelerationReward = 0;
            RuntimeRewards.TotalLastAccumulatedReward = 0;
            RuntimeRewards.TotalLoopSpentTimeReward = 0;
            RuntimeRewards.TotalPassCheckpointReward = 0;
            RuntimeRewards.TotalSpeedReward = 0;
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
            m_stuckedFrames = 0;
            m_collidedFrames = 0;
            m_wrongDirectionFrames = 0;
            m_rightDirectionFrames = 0;
            m_quickStopPenalty = 0;
        }

        private void SetDecisionRequester()
        {
            var decisionRequester = GetComponent<DecisionRequester>();
            //TODO set your decision requester
            // decisionRequester.DecisionPeriod
            // decisionRequester.TakeActionsBetweenDecisions
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
            Sensors = new Sensor[3];
            Sensors[0] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/0"),
                HitValidationDistance = 2f,
                RayDistance = 30
            };
            Sensors[1] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/90"),
                HitValidationDistance = 0.8f,
                RayDistance = 10
            };
            Sensors[2] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/-90"),
                HitValidationDistance = 0.8f,
                RayDistance = 10
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

                    //m_LastHitCheckpointIndex = Random.Range(0, Checkpoints.Length - 1);
                    m_LastHitCheckpointIndex = (m_LastHitCheckpointIndex - 2 + Checkpoints.Length) % Checkpoints.Length;
                    m_lastLeftCheckpointIndex = -1;
                    InitCheckpointIndex = m_LastHitCheckpointIndex;
                    m_targetingCheckpointIndex = (m_LastHitCheckpointIndex + 1) % Checkpoints.Length;
                    var collider = Checkpoints[m_LastHitCheckpointIndex];
                    transform.localRotation
                        = collider.transform.rotation;
                    if (Vector3.Dot(transform.forward, Checkpoints[m_targetingCheckpointIndex].transform.forward) < 0)
                    {
                        transform.localRotation
                                        = collider.transform.rotation * Quaternion.Euler(
                                            0,
                                                    180,
                                            0);
                    }

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
            {
                m_LastHitCheckpointIndex = InitCheckpointIndex;
                m_lastLeftCheckpointIndex = -1;
            }

        }
        #endregion Initialization

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
            int nextCheckpointIndex = (m_LastHitCheckpointIndex + 1) % Checkpoints.Length;
            Debug.DrawLine(AgentSensorTransform.position, Checkpoints[m_targetingCheckpointIndex].transform.position, Color.green);
            Debug.DrawLine(AgentSensorTransform.position, Checkpoints[nextCheckpointIndex].transform.position, Color.cyan);
            Debug.DrawLine(AgentSensorTransform.position, Checkpoints[m_LastHitCheckpointIndex].transform.position, Color.magenta);
            // Add your rewards/penalties
            for (int i = 0; i < Sensors.Length; i++)
            {
                if (i == 0 || i == 1 || i == 5 || i == 6)
                {
                    continue;
                }



                var current = Sensors[i];
                var xform = current.Transform;
                for (int j = 0; j < 3; j++)
                {
                    float angle = -10 + j * 10;
                    var hit = Physics.Raycast(AgentSensorTransform.position, (Quaternion.AngleAxis(angle, Vector3.forward) * xform.forward).normalized, out var hitInfo, m_raycastMaxDistance, Mask, QueryTriggerInteraction.Ignore);
                    var endPoint = AgentSensorTransform.position + (Quaternion.AngleAxis(angle, Vector3.forward) * xform.forward).normalized * m_raycastMaxDistance;
                    bool drawKartRadius = true;
                    Color color = Color.blue;
                    if (hit)
                    {
                        if (hitInfo.distance <= m_collisionRadius)
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
                        Debug.DrawLine(AgentSensorTransform.position, AgentSensorTransform.position + (Quaternion.AngleAxis(angle, Vector3.up) * xform.forward).normalized * m_collisionRadius, Color.yellow);
                    }
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
                        var checkpoint = Checkpoints[m_LastHitCheckpointIndex].transform;
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
            if (Mode == AgentMode.Inferencing)
            {
                return;
            }
            DrawDetectingRays();

            UpdateKartStatusRewards();

            RuntimeRewards.OverallRewards
                   = RuntimeRewards.TotalSpeedReward
                   + RuntimeRewards.TotalTowardsCheckpointReward
                   + RuntimeRewards.TotalPassCheckpointReward
                   + RuntimeRewards.TotalAccelerationReward
                   + RuntimeRewards.TotalLoopSpentTimeReward
                   + RuntimeRewards.TotalLastAccumulatedReward
                   + RuntimeRewards.TotalFinishCheckpointAverageSpeedReward
                   + RuntimeRewards.TotalStuckPenalty
                   + RuntimeRewards.TotalStayCollisionPenalty
                   + RuntimeRewards.TotalTriggerCollisionPenalty;
            if (m_EndEpisode)
            {
                m_EndEpisode = false;

                Debug.Log($"[Episode Ended] [FinalRewards]:{RuntimeRewards.OverallRewards}");
                Debug.Log($"[Episode Ended],[Penalty] " +
                    $"Stuck:{RuntimeRewards.TotalStuckPenalty}, " +
                    $"StayCollision:{RuntimeRewards.TotalStayCollisionPenalty}, " +
                    $"TriggerCollision:{RuntimeRewards.TotalTriggerCollisionPenalty}." +
                    $"QuickStopPenalty:{m_quickStopPenalty}.");
                Debug.Log($"[Episode Ended], [Reward]" +
                   $"Speed:{RuntimeRewards.TotalSpeedReward}, " +
                   $"TowardsCheckpoint:{RuntimeRewards.TotalTowardsCheckpointReward}, " +
                   $"PassCheckpoint:{RuntimeRewards.TotalPassCheckpointReward}, " +
                   $"Acceleration:{RuntimeRewards.TotalAccelerationReward}, " +
                   $"LoopSpentTime:{RuntimeRewards.TotalLoopSpentTimeReward}, " +
                   $"LastAccumulate:{RuntimeRewards.TotalLastAccumulatedReward},  " +
                   $"FinishCheckpointAverageSpeedReward:{RuntimeRewards.TotalFinishCheckpointAverageSpeedReward}");
                EndEpisode();
            }
            Speed = m_Kart.LocalSpeed();

        }

        #region Collisions

        private void OnCollisionEnter(Collision collision)
        {
            float speed = m_Kart.LocalSpeed();
            Debug.Log($"[OnCollisionEnter],kart speed:{speed}, kart forward:{m_Kart.transform.forward}");
            for (int i = 0; i < collision.contactCount; i++)
            {
                var contact = collision.GetContact(i);
                if (contact.normal.y != 0)
                {
                    return;
                }
            }
            Collided = true;
            //for (int i = 0; i < collision.contactCount; i++)
            //{
            //    var contact = collision.GetContact(i);
            //    var direction = (contact.point - transform.position).normalized;
            //    float speedFactor = 0;
            //    if (speed > 0f)
            //    {
            //        speedFactor = 1f;
            //    }
            //    else if (speed < 0f)
            //    {
            //        speedFactor = -1f;
            //    }
            //    var factor = Vector3.Dot(direction, m_Kart.Rigidbody.velocity);
            //    if (factor > 0f)
            //    {
            //        var RewardsConfig.LastAccumulatedReward * 100000 = factor * RewardsConfig.TriggerCollisionPenalty;
            //        AddReward(penalty, ref RuntimeRewards.TotalTriggerCollisionPenalty);
            //        m_deltaCollisionPenalty += penalty;
            //    }
            //}
            m_collidedFrames++;
            //m_checkPosition = transform.position;
        }
        private void OnCollisionStay(Collision collision)
        {
            float speed = m_Kart.LocalSpeed();
            Debug.Log($"[OnCollisionStay],kart speed:{speed}, kart velocity:{m_Kart.Rigidbody.velocity}");
            for (int i = 0; i < collision.contactCount; i++)
            {
                var contact = collision.GetContact(i);
                if (contact.normal.y != 0)
                {
                    return;
                }
            }
            //for (int i = 0; i < collision.contactCount; i++)
            //{
            //    var contact = collision.GetContact(i);
            //    var direction = (contact.point - transform.position).normalized;
            //    //if (i < 3)
            //    //{
            //    //    m_contactPoints[i] = contact.point;
            //    //}
            //    //float speedFactor = 0;
            //    //if (speed > 0f)
            //    //{
            //    //    speedFactor = 1f;
            //    //}
            //    //else if (speed < 0f)
            //    //{
            //    //    speedFactor = -1f;
            //    //}
            //    //var factor = Vector3.Dot(direction, m_Kart.transform.forward * speed);
            //    //var factor2 = Vector3.Dot(direction, m_Kart.transform.forward);
            //    //AddReward((factor + factor2) * RewardsConfig.StayCollisionPenalty, ref RuntimeRewards.TotalStayCollisionPenalty);
            //    //m_deltaCollisionPenalty += (factor + factor2) * RewardsConfig.StayCollisionPenalty;
            //}
            AddReward(RewardsConfig.StayCollisionPenalty, ref RuntimeRewards.TotalStayCollisionPenalty);

            //m_deltaCollisionPenalty += RewardsConfig.StayCollisionPenalty;
            //AddReward(RewardsConfig.StayCollisionPenalty * m_collidedFrames, ref RuntimeRewards.TotalStayCollisionPenalty);
            m_collidedFrames++;
            if (m_collidedFrames > TrainingConfigInfo.MaxCollidedFrames)
            {
                Debug.Log("[Quick Stop Training Episode Ended] MaxCollidedFrames Exceeded");
                m_EndEpisode = true;
                AddReward(TrainingConfigInfo.QuickStopPenalty_Collision, ref m_quickStopPenalty);
            }
            //m_checkPosition = transform.position;
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
                //int nextCheckpointIndex = (m_CheckpointIndex + 1) % Checkpoints.Length;
                //if (index==m_CheckpointIndex)
                //{

                //}
                //var directionDot = Vector3.Dot(m_Kart.Rigidbody.velocity.normalized, other.transform.forward);
                //AddReward(RewardsConfig.PassCheckpointReward * directionDot, ref RuntimeRewards.TotalPassCheckpointReward);


                //bool rightDirection = directionDot > 0;
                //if (!rightDirection)
                //{
                //    m_targetingCheckpointIndex = index;
                //    AddReward(-RewardsConfig.PassCheckpointReward, ref RuntimeRewards.TotalPassCheckpointReward);
                //}
                //else
                //{
                //if (index == m_lastLeftCheckpointIndex)
                //{
                //    m_targetingCheckpointIndex = index;
                //    AddReward(-RewardsConfig.PassCheckpointReward, ref RuntimeRewards.TotalPassCheckpointReward);
                //}
                //else
                //{
                m_targetingCheckpointIndex = (index + 1) % Checkpoints.Length;
                //AddReward(RewardsConfig.PassCheckpointReward, ref RuntimeRewards.TotalPassCheckpointReward);
                //}
                m_lastLeftCheckpointIndex = index;

                //}
            }
        }

        void OnTriggerEnter(Collider other)
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


                if (index == -1)
                {
                    return;
                }
                // specify targeting CheckpointIndex

                Debug.Log($"[Enter Checkpoint] index:{index}, m_CheckpointIndex:{m_LastHitCheckpointIndex},m_targetingCheckpointIndex:{m_targetingCheckpointIndex}");

                if (index == m_targetingCheckpointIndex)
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
                        if (m_loopCount >= TrainingConfigInfo.EpisodeMaxLoopCount)
                        {
                            Debug.Log("[Stop Training Episode Ended] Rich max loop");
                            m_EndEpisode = true;
                        }
                    }
                }
                m_targetingCheckpointIndex = (index + 1) % Checkpoints.Length;
                m_LastHitCheckpointIndex = index;
                //else // go backtoward 
                //{
                //    // paneltly according to checkpoint index diff
                //    //int backwardCount = (index - nextCheckpointIndex - Checkpoints.Length) % Checkpoints.Length;
                //    int backwardCount = -1;
                //    AddReward(RewardsConfig.PassCheckpointReward * backwardCount, ref RuntimeRewards.TotalPassCheckpointReward);
                //}
            }
        }

        #endregion Collisions

        #endregion UpdateStatus

        #region Reward Strategy
        private void UpdateKartStatusRewards()
        {
            //average of next to checkpoint direction rewards
            var nextCollider = Checkpoints[m_targetingCheckpointIndex];
            var direction1 = (nextCollider.transform.position - m_Kart.transform.position).normalized;
            var reward = Vector3.Dot(transform.forward * m_Kart.LocalSpeed(), direction1);
            if (!Collided)
            {
                if (reward < 0)
                {
                    m_wrongDirectionFrames++;
                    m_rightDirectionFrames = 0;
                    //AddReward(reward21 * RewardsConfig.SpeedTowardNextCheckpointReward * m_wrongDirectionFrames, ref RuntimeRewards.TotalSpeedTowardNextCheckpointReward);

                }
                else
                {
                    m_rightDirectionFrames++;
                    m_wrongDirectionFrames = 0;
                    //AddReward(reward21 * RewardsConfig.SpeedTowardNextCheckpointReward, ref RuntimeRewards.TotalSpeedTowardNextCheckpointReward);
                }
                AddReward(reward * RewardsConfig.TowardsCheckpointReward, ref RuntimeRewards.TotalTowardsCheckpointReward);

            }

            AddReward(m_Kart.LocalSpeed() * RewardsConfig.SpeedReward, ref RuntimeRewards.TotalSpeedReward);

            // Add rewards if the agent is heading in the right direction
            AddReward((m_Acceleration && !m_Brake ? 1.0f : 0.0f) * RewardsConfig.AccelerationReward, ref RuntimeRewards.TotalAccelerationReward);
            //AddReward(m_Kart.LocalSpeed() * RewardsConfig.SpeedReward, ref RuntimeRewards.TotalSpeedReward);

            AddReward(RewardsConfig.LastAccumulatedReward, ref RuntimeRewards.TotalLastAccumulatedReward);

            CheckStucked();

            if (m_wrongDirectionFrames >= TrainingConfigInfo.MaxWrongDirectionFrames)
            {
                Debug.Log("[Quick Stop Training Episode Ended] MaxWrongDirectionFrames Exceeded");
                m_EndEpisode = true;
                AddReward(TrainingConfigInfo.QuickStopPenalty_WrongDirection, ref m_quickStopPenalty);
            }
            if (m_rightDirectionFrames > TrainingConfigInfo.MaxRightDirectionFrames && (TrainingConfigInfo.MaxRightDirectionFrames > 0))
            {
                Debug.Log("[Quick Stop Training Episode Ended] MaxRightDirectionFrames Exceeded");
                m_EndEpisode = true;
            }
        }
        private void CheckStucked()
        {
            float moved = (transform.position - m_checkPosition).magnitude;
            if (Stucked)
            {
                m_stuckedFrames++;

                if (m_stuckedFrames > (TrainingConfigInfo.MaxStuckedFrames + TrainingConfigInfo.StuckedFramesThreshold))
                {
                    Debug.Log("[Quick Stop Training Episode Ended] MaxStuckedFrames Exceeded");
                    m_EndEpisode = true;
                    AddReward(TrainingConfigInfo.QuickStopPenalty_Stucked, ref m_quickStopPenalty);
                }

                // escape from stuck status
                if (moved > TrainingConfigInfo.EscapeStuckedDistance)
                {
                    Stucked = false;
                    m_checkPosition = transform.position;
                    m_stuckedFrames = 0;
                }

            }
            else
            {
                if (moved < TrainingConfigInfo.DistanceThresholdStucked)
                {
                    m_stuckedFrames++;
                    if (m_stuckedFrames > TrainingConfigInfo.StuckedFramesThreshold)
                    {
                        Stucked = true;
                    }
                }
                else
                {
                    m_checkPosition = transform.position;
                }
            }
            //if (Stucked)
            //{
            //    var direction = (m_checkPosition - transform.position).normalized;

            //    var factor = Vector3.Dot(direction, m_Kart.Rigidbody.velocity / m_Kart.GetMaxSpeed());
            //    AddReward(factor * RewardsConfig.StuckPenalty, ref RuntimeRewards.TotalStuckPenalty);
            //}

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

        public override void OnEpisodeBegin()
        {
            InitialiseResetParameters();
            InitializeSenses();
            switch (Mode)
            {
                case AgentMode.Training:
                    m_CheckpointIndex = Random.Range(0, Checkpoints.Length - 1);
                    var collider = Checkpoints[m_CheckpointIndex];
                    transform.localRotation = collider.transform.rotation;
                    transform.position = collider.transform.position;
                    m_Kart.Rigidbody.velocity = default;
                    m_Acceleration = false;
                    m_Brake = false;
                    m_Steering = 0f;
                    break;
            }
        }

        void OnTriggerEnter(Collider other)
        {
            // TODO implement what should the agent do when it touched a checkpoint
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            // TODO Add your observations
            // For example
            sensor.AddObservation(m_Kart.LocalSpeed());
            foreach (var current in Sensors)
            {
                var xform = current.Transform;
                var hit = Physics.Raycast(AgentSensorTransform.position, xform.forward, out var hitInfo,
                    current.RayDistance, Mask, QueryTriggerInteraction.Ignore);
                var hit1 = Physics.Raycast(AgentSensorTransform.position, Quaternion.AngleAxis(10, Vector3.forward) * xform.forward, out var hitInfo1,
                   current.RayDistance, Mask, QueryTriggerInteraction.Ignore);
                var hit2 = Physics.Raycast(AgentSensorTransform.position, Quaternion.AngleAxis(-10, Vector3.forward) * xform.forward, out var hitInfo2,
                    current.RayDistance, Mask, QueryTriggerInteraction.Ignore);
                sensor.AddObservation(hit ? hitInfo.distance : current.RayDistance);
                sensor.AddObservation(hit1 ? hitInfo1.distance : current.RayDistance);
                sensor.AddObservation(hit2 ? hitInfo2.distance : current.RayDistance);
            }
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            base.OnActionReceived(actions);
            InterpretDiscreteActions(actions);

            // TODO Add your rewards/penalties
        }

        private void InterpretDiscreteActions(ActionBuffers actions)
        {
            m_Steering = actions.DiscreteActions[0] - 1f;
            m_Acceleration = actions.DiscreteActions[1] >= 1.0f;
            m_Brake = actions.DiscreteActions[1] < 1.0f;
        }
    }
}
