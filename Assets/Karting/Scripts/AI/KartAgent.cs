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
        private int m_CheckpointIndex;

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
            // behaviorParameters.BehaviorType
            // behaviorParameters.BehaviorName
            // behaviorParameters.BrainParameters.VectorObservationSize
            // behaviorParameters.BrainParameters.NumStackedVectorObservations
            // behaviorParameters.BrainParameters.ActionSpec   
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

        private void Start()
        {
            // If the agent is training, then at the start of the simulation, pick a random checkpoint to train the agent.
            OnEpisodeBegin();

            if (Mode == AgentMode.Inferencing)
                m_CheckpointIndex = InitCheckpointIndex;
        }

        private void Update()
        {
            if (m_EndEpisode)
            {
                m_EndEpisode = false;
                AddReward(m_LastAccumulatedReward);
                EndEpisode();
                OnEpisodeBegin();
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

                sensor.AddObservation(hit ? hitInfo.distance : current.RayDistance);
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
