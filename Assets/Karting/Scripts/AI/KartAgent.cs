using KartGame.KartSystems;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;
using Random = UnityEngine.Random;
using System;
using Unity.MLAgents.Policies;
using static Codice.CM.Common.CmCallContext;

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
        [Header("Rewards")]
        [Tooltip("What penalty is given when the agent crashes?")]
        public float Penalty = -3;

        [Tooltip("How much reward is given when the agent successfully passes the checkpoints?")]
        public float PassCheckpointReward = 1f;

        [Tooltip("Should typically be a small value, but we reward the agent for moving in the right direction.")]
        public float TowardsCheckpointReward = 0.01f;

        [Tooltip("Typically if the agent moves faster, we want to reward it for finishing the track quickly.")]
        public float SpeedReward = 0.005f;

        [Tooltip("Reward the agent when it keeps accelerating")]
        public float AccelerationReward = 0.1f;
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
        private int m_CheckpointIndex;

        private bool m_EndEpisode;
        private float m_LastAccumulatedReward;
        private float m_startTime = 0;
        private void Awake()
        {
            m_Kart = GetComponent<ArcadeKart>();

            AgentSensorTransform = transform.Find("MLAgent_Sensors");
            SetBehaviorParameters();
            SetDecisionRequester();
            Debug.Log("=====Awake====");
            Debug.Log(m_Kart.transform.position);
        }

        private void SetBehaviorParameters()
        {
            var behaviorParameters = GetComponent<BehaviorParameters>();
            behaviorParameters.UseChildSensors = true;
            behaviorParameters.ObservableAttributeHandling = ObservableAttributeOptions.Ignore;
            // TODO set other behavior parameters according to your own agent implementation, especially following ones:
            behaviorParameters.BehaviorType = BehaviorType.Default;
            behaviorParameters.BehaviorName = "First-SG-Driver";
            behaviorParameters.BrainParameters.VectorObservationSize = 10; // size of the ML input data, should be the same as the `AddObservation` number
            behaviorParameters.BrainParameters.NumStackedVectorObservations = 4;
            behaviorParameters.BrainParameters.ActionSpec = ActionSpec.MakeDiscrete(3, 2); // format of the ML model output data [0, 1, 2] [0, 1, 2]

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
                RayDistance = 5
            };
            Debug.Log(Sensors[0]);
            Sensors[1] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/-60"),
                HitValidationDistance = 1.5f,
                RayDistance = 10
            };
            Sensors[2] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/-30"),
                HitValidationDistance = 1.8f,
                RayDistance = 15
            };
            Sensors[3] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/0"),
                HitValidationDistance = 3f,
                RayDistance = 30
            };
            Sensors[4] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/30"),
                HitValidationDistance = 1.8f,
                RayDistance = 15
            };
            Sensors[5] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/60"),
                HitValidationDistance = 1.5f,
                RayDistance = 15,

            };
            Sensors[6] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/90"),
                HitValidationDistance = 1f,
                RayDistance = 10
            };
            Sensors[7] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/45"),
                HitValidationDistance = 1.7f,
                RayDistance = 12.5f
            };
            Sensors[8] = new Sensor
            {
                Transform = transform.Find("MLAgent_Sensors/-45"),
                HitValidationDistance = 1.7f,
                RayDistance = 12.5f
            };

            // set Mask
            Mask = LayerMask.GetMask("Default", "Ground", "Environment", "Track");

            // set Checkpoints
            Checkpoints = GameObject.Find("Checkpoints").transform.GetComponentsInChildren<Collider>();

            // set CheckpointMask
            CheckpointMask = LayerMask.GetMask("CompetitionCheckpoints", "TrainingCheckpoints");
        }

        private void Start()
        {
            // If the agent is training, then at the start of the simulation, pick a random checkpoint to train the agent.
            OnEpisodeBegin();

            if (Mode == AgentMode.Inferencing)
                m_CheckpointIndex = InitCheckpointIndex;
        }


        private void LateUpdate()
        {
            Debug.Log("Late Update detected.");
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
            if (Mode == AgentMode.Inferencing)
            {
                return;
            }
            if (m_EndEpisode)
            {
                m_EndEpisode = false;
                EndEpisode();
            }

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
                    1f, Mask, QueryTriggerInteraction.Ignore);
                Debug.DrawLine(AgentSensorTransform.position, AgentSensorTransform.position + xform.forward * 5, Color.blue);
                if (hit)
                {
                    Debug.DrawLine(AgentSensorTransform.position, hitInfo.point, Color.red);
                    m_EndEpisode = true;
                    AddReward(Penalty);
                }
            }
        }

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
            m_Steering = 0f;
            InitialiseResetParameters();
            InitializeSenses();
            switch (Mode)
            {
                case AgentMode.Training:
                    m_CheckpointIndex = Random.Range(0, Checkpoints.Length - 1);
                    InitCheckpointIndex = m_CheckpointIndex;
                    var collider = Checkpoints[m_CheckpointIndex];

                    transform.localRotation = collider.transform.rotation;
                    transform.position = collider.transform.position;
                    m_Kart.Rigidbody.velocity = default;
                    m_Acceleration = false;
                    m_Brake = false;
                    break;
            }
            m_startTime = Time.time;
        }

        void OnTriggerEnter(Collider other)
        {
            // TODO implement what should the agent do when it touched a checkpoint
            var maskedValue = 1 << other.gameObject.layer;
            var triggered = maskedValue & CheckpointMask;

            FindCheckpointIndex(other, out var index);

            // Ensure that the agent touched the checkpoint and the new index is greater than the m_CheckpointIndex.
            if (triggered > 0 && index > m_CheckpointIndex || index == 0 && m_CheckpointIndex == Checkpoints.Length - 1)
            {
                AddReward(PassCheckpointReward);
                if (index == InitCheckpointIndex)
                {
                    // a loop
                    float spentTime = Time.time - m_startTime;
                    Debug.Log($"Finish a loop, spend time:{spentTime}");
                    m_startTime = Time.time;
                    AddReward(100 / spentTime);
                    // TODO: award according to spent time
                }
                m_CheckpointIndex = index;
            }

            if (triggered <= 0)
            {
                Debug.Log("Penalty!!!");
                AddReward(Penalty);
            }
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

            foreach (var current in Sensors)
            {
                var xform = current.Transform;
                var hit = Physics.Raycast(AgentSensorTransform.position, xform.forward, out var hitInfo,
                    current.RayDistance, Mask, QueryTriggerInteraction.Ignore);

                sensor.AddObservation(hit ? hitInfo.distance : current.RayDistance);
            }

            //Vector3 agentPosition = m_Kart.transform.position;
            //sensor.AddObservation(agentPosition);

            //// Find all nearby objects within a certain radius
            //float radius = 5f;
            //Collider[] colliders = Physics.OverlapSphere(agentPosition, radius);

            //// Loop through the colliders and add relevant observations
            //foreach (Collider collider in colliders)
            //{
            //    // Check if the collider belongs to a collectible object
            //    if (collider.CompareTag("Collectible"))
            //    {
            //        // Get the position of the collectible
            //        Vector3 collectiblePosition = collider.transform.position;

            //        // Add the collectible's position as an observation
            //        sensor.AddObservation(collectiblePosition);
            //    }
            //}

            //Add an observation for direction of the agent to the next checkpoint.

            //var next = (m_CheckpointIndex + 1) % Checkpoints.Length;
            //var nextCollider = Checkpoints[next];
            //if (nextCollider == null)
            //    return;

            //var direction = (nextCollider.transform.position - m_Kart.transform.position).normalized;
            //sensor.AddObservation(Vector3.Dot(m_Kart.Rigidbody.velocity.normalized, direction));
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            base.OnActionReceived(actions);
            InterpretDiscreteActions(actions);


            //Find the next checkpoint when registering the current checkpoint that the agent has passed.
            var next = (m_CheckpointIndex + 1) % Checkpoints.Length;
            var nextCollider = Checkpoints[next];
            var direction = (nextCollider.transform.position - m_Kart.transform.position).normalized;
            var reward = Vector3.Dot(m_Kart.Rigidbody.velocity.normalized, direction);

            // Add rewards if the agent is heading in the right direction
            AddReward(reward * TowardsCheckpointReward);
            AddReward((m_Acceleration && !m_Brake ? 1.0f : 0.0f) * AccelerationReward);
            AddReward(m_Kart.LocalSpeed() * SpeedReward);
        }

        private void InterpretDiscreteActions(ActionBuffers actions)
        {
            m_Steering = actions.DiscreteActions[0] - 1f;
            m_Acceleration = actions.DiscreteActions[1] >= 1.0f;
            m_Brake = actions.DiscreteActions[1] < 1.0f;
        }

        //private float RightDirection()
        //{
        //    // Find the next checkpoint when registering the current checkpoint that the agent has passed.
        //    var next = (m_CheckpointIndex + 1) % Checkpoints.Length;
        //    var nextCollider = Checkpoints[next];
        //    if (nextCollider != null)
        //    {
        //        var direction = (nextCollider.transform.position - m_Kart.transform.position).normalized;
        //        var reward = Vector3.Dot(m_Kart.Rigidbody.velocity.normalized, direction);
        //        return reward;
        //    }
        //    else
        //    {
        //        return 0f;
        //    }
        //}
    }
}